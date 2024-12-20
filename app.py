import streamlit as st
import re
import pickle


import os

from dotenv import load_dotenv
import getpass
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def save_display_tabs(display_tabs, company_name):
    with open(f'display_tabs_{company_name}.pkl', 'wb') as f:
        pickle.dump(display_tabs, f)
def _set_env(var: str):
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var + ":")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


def sanitize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

async def compile_investment_memo(company):
    from langchain_openai import ChatOpenAI

    fast_llm = ChatOpenAI(model="gpt-4o")
    # Uncomment for a Fireworks model
    # fast_llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v1", max_tokens=32_000)
    long_context_llm = ChatOpenAI(model="gpt-4o")

    from typing import List, Optional, Any

    from langchain_core.prompts import ChatPromptTemplate

    from pydantic import BaseModel, Field

    class IntermediateResults:
        def __init__(self):
            self.initial_outline=""
            self.related_subjects=""
            self.perspectives: Any
            self.queries: Any
            self.generated: Any
            self.cited_references: Any
            self.cited_urls: Any
            self.formatted_message: Any
            self.refined_outline = ""
            self.interview_results: Any
            self.results: Any
            self.updated_outline: Any


    display_tabs = IntermediateResults()

    direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a private equity analyst. Create a comprehensive due diligence outline about a user-provided company or investment opportunity. Include sections on market analysis, competitive landscape, financial performance, detailed SWOT analysis, risks, and potential returns.",
            ),
            ("user", "{topic}"),
        ]
    )

    class Subsection(BaseModel):
        subsection_title: str = Field(..., title="Title of the subsection")
        description: str = Field(..., title="Content of the subsection")

        @property
        def as_str(self) -> str:
            return f"### {self.subsection_title}\n\n{self.description}".strip()

    class Section(BaseModel):
        section_title: str = Field(..., title="Title of the section")
        description: str = Field(..., title="Content of the section")
        subsections: Optional[List[Subsection]] = Field(
            default=None,
            title="Titles and descriptions for each subsection of the Investment memo.",
        )

        @property
        def as_str(self) -> str:
            subsections = "\n\n".join(
                f"### {subsection.subsection_title}\n\n{subsection.description}"
                for subsection in self.subsections or []
            )
            return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()

    class Outline(BaseModel):
        page_title: str = Field(..., title="Title of the Investment memo")
        sections: List[Section] = Field(
            default_factory=list,
            title="Titles and descriptions for each section of the Investment memo.",
        )

        @property
        def as_str(self) -> str:
            sections = "\n\n".join(section.as_str for section in self.sections)
            return f"# {self.page_title}\n\n{sections}".strip()

    generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(
        Outline
    )

    example_topic = f"Evaluating a potential investment in {company}, a startup FinTech firm."

    initial_outline = generate_outline_direct.invoke({"topic": example_topic})
    display_tabs.initial_outline = initial_outline.as_str



    gen_related_topics_prompt = ChatPromptTemplate.from_template(
        """I'm writing an Investment memo for a company and topic mentioned below. Please identify and recommend some Investment opportunity web pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this company, topic and sector, or examples that help me understand the typical content and structure included in investment theses for similar companies and opportunities.

    Please list the as many subjects and urls as you can.

    Topic of interest: {topic}
    """
    )

    class RelatedSubjects(BaseModel):
        topics: List[str] = Field(
            description="Comprehensive list of related subjects as background research.",
        )

    expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(
        RelatedSubjects
    )

    related_subjects = await expand_chain.ainvoke({"topic": example_topic})
    display_tabs.related_subjects = related_subjects


    class Editor(BaseModel):
        affiliation: str = Field(
            description="Primary affiliation of the editor.",
        )
        name: str = Field(
            description="Name of the editor.",
            pattern=r"^[A-Za-z0-9 _-]{1,64}$"  # allow spaces now
        )

        role: str = Field(
            description="Role of the editor in the context of the topic.",
        )
        description: str = Field(
            description="Description of the editor's focus, concerns, and motives.",
        )

        @property
        def persona(self) -> str:
            return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    class Perspectives(BaseModel):
        editors: List[Editor] = Field(
            description="Comprehensive list of editors with their roles and affiliations.",
            # Add a pydantic validation/restriction to be at most M editors
        )

    gen_perspectives_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You need to select a diverse (and distinct) group of Investment analysts who will work together to create a comprehensive Investment memo on the topic. Each of them represents a different perspective, role, or affiliation related to this company, investment opportunity and topic.\
        You can use other Investment opportunity pages of related topics for inspiration. For each editor, add a description of what they will focus on.

        Investment memo outlines of related topics for inspiration:
        {examples}""",
            ),
            ("user", "Topic of interest: {topic}"),
        ]
    )

    gen_perspectives_chain = gen_perspectives_prompt | ChatOpenAI(
        model="gpt-3.5-turbo"
    ).with_structured_output(Perspectives)

    from langchain_community.retrievers import WikipediaRetriever
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables import chain as as_runnable

    wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)

    def format_doc(doc, max_length=1000):
        related = "- ".join(doc.metadata["categories"])
        return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
               :max_length
               ]

    def format_docs(docs):
        return "\n\n".join(format_doc(doc) for doc in docs)

    @as_runnable
    async def survey_subjects(topic: str):
        related_subjects = await expand_chain.ainvoke({"topic": topic})
        retrieved_docs = await wikipedia_retriever.abatch(
            related_subjects.topics, return_exceptions=True
        )
        all_docs = []
        for docs in retrieved_docs:
            if isinstance(docs, BaseException):
                continue
            all_docs.extend(docs)
        formatted = format_docs(all_docs)
        return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})

    perspectives = await survey_subjects.ainvoke(example_topic)
    display_tabs.perspectives = perspectives.model_dump()


    from typing import Annotated

    from langchain_core.messages import AnyMessage
    from typing_extensions import TypedDict

    from langgraph.graph import END, StateGraph, START

    def add_messages(left, right):
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        return left + right

    def update_references(references, new_references):
        if not references:
            references = {}
        references.update(new_references)
        return references

    def update_editor(editor, new_editor):
        # Can only set at the outset
        if not editor:
            return new_editor
        return editor

    class InterviewState(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]
        references: Annotated[Optional[dict], update_references]
        editor: Annotated[Optional[Editor], update_editor]

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.prompts import MessagesPlaceholder

    gen_qn_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced Private Equity Investment manager conducting due diligence on a potential investment, and want to contribute to a specific section of the Investment memo. \
    Besides your identity as a Private Equity Investor, you have a specific focus when researching this company, opprtunity, sector and topic. \
    Now, you are chatting with an industry expert to get information. Ask one question at a time about the target company, 
            focusing on areas like market trends, competitive landscape, unit economics, technology differentiation, 
            and regulatory environment.

    When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
    Please only ask one question at a time and don't ask what you have asked before.\
    Your questions should be related to the company, investment thesis and topic you want to write.
    Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

    Stay true to your specific perspective:

    {persona}""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    def tag_with_name(ai_message: AIMessage, name: str):
        ai_message.name = sanitize_name(name)
        return ai_message

    def swap_roles(state: InterviewState, name: str):
        converted = []
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.name != name:
                message = HumanMessage(**message.model_dump(exclude={"type"}))
            converted.append(message)
        return {"messages": converted}

    @as_runnable
    async def generate_question(state: InterviewState):
        editor = state["editor"]
        gn_chain = (
                RunnableLambda(swap_roles).bind(name=sanitize_name(editor.name))
                | gen_qn_prompt.partial(persona=editor.persona)
                | fast_llm
                | RunnableLambda(tag_with_name).bind(name=sanitize_name(editor.name))
        )
        result = await gn_chain.ainvoke(state)
        return {"messages": [result]}

    messages = [
        HumanMessage(f"So you said you were writing an article on {example_topic}?")
    ]
    question = await generate_question.ainvoke(
        {
            "editor": perspectives.editors[0],
            "messages": messages,
        }
    )



    class Queries(BaseModel):
        queries: List[str] = Field(
            description="Comprehensive list of search engine queries to answer the user's questions.",
        )

    gen_queries_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Query the search engine to answer the user's questions.",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    gen_queries_chain = gen_queries_prompt | ChatOpenAI(
        model="gpt-3.5-turbo"
    ).with_structured_output(Queries, include_raw=True)

    queries = await gen_queries_chain.ainvoke(
        {"messages": [HumanMessage(content=question["messages"][0].content)]}
    )
    display_tabs.queries=queries["parsed"].queries


    class AnswerWithCitations(BaseModel):
        answer: str = Field(
            description="Comprehensive answer to the user's question with citations.",
        )
        cited_urls: List[str] = Field(
            description="List of urls cited in the answer.",
        )

        @property
        def as_str(self) -> str:
            return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
                f"[{i + 1}]: {url}" for i, url in enumerate(self.cited_urls)
            )

    gen_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert who can use information effectively. You are chatting with a Private Equity investment expert who wants\
     to write an Investment memo on the topic you know. You have gathered the related information and will now use the information to form a response.

    Make your response as informative as possible and make sure every sentence is supported by the gathered information.
    Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
        AnswerWithCitations, include_raw=True
    ).with_config(run_name="GenerateAnswer")

    from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
    from langchain_core.tools import tool

    from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
    from langchain_core.tools import tool

    # '''
    # # Tavily is typically a better search engine, but your free queries are limited
    # search_engine = TavilySearchResults(max_results=4)
    #
    # @tool
    # async def search_engine(query: str):
    #     """Search engine to the internet."""
    #     results = tavily_search.invoke(query)
    #     return [{"content": r["content"], "url": r["url"]} for r in results]
    #
    # '''
    # DDG
    search_engine = DuckDuckGoSearchAPIWrapper()

    @tool
    async def search_engine(query: str):
        """Search engine to the internet."""
        results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in results]

    # DDG

    @tool
    async def search_engine(query: str):
        """Search engine to the internet."""
        results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in results]

    import json

    from langchain_core.runnables import RunnableConfig

    async def gen_answer(
            state: InterviewState,
            config: Optional[RunnableConfig] = None,
            name: str = "Subject_Matter_Expert",
            max_str_len: int = 15000,
    ):
        swapped_state = swap_roles(state, name)  # Convert all other AI messages
        queries = await gen_queries_chain.ainvoke(swapped_state)
        query_results = await search_engine.abatch(
            queries["parsed"].queries, config, return_exceptions=True
        )
        successful_results = [
            res for res in query_results if not isinstance(res, Exception)
        ]
        all_query_results = {
            res["url"]: res["content"] for results in successful_results for res in results
        }
        # We could be more precise about handling max token length if we wanted to here
        dumped = json.dumps(all_query_results)[:max_str_len]
        ai_message: AIMessage = queries["raw"]
        tool_call = queries["raw"].tool_calls[0]
        tool_id = tool_call["id"]
        tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
        swapped_state["messages"].extend([ai_message, tool_message])
        # Only update the shared state with the final answer to avoid
        # polluting the dialogue history with intermediate messages
        generated = await gen_answer_chain.ainvoke(swapped_state)
        cited_urls = set(generated["parsed"].cited_urls)
        # Save the retrieved information to a the shared state for future reference
        cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
        formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
        display_tabs.generated=generated
        display_tabs.cited_references=cited_references
        display_tabs.cited_urls=cited_urls
        display_tabs.formatted_message=formatted_message
        return {"messages": [formatted_message], "references": cited_references}

    example_answer = await gen_answer(
        {"messages": [HumanMessage(content=question["messages"][0].content)]}
    )


    max_num_turns = 5
    from langgraph.pregel import RetryPolicy

    def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"

    builder = StateGraph(InterviewState)

    builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=5))
    builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=5))
    builder.add_conditional_edges("answer_question", route_messages)
    builder.add_edge("ask_question", "answer_question")

    builder.add_edge(START, "ask_question")
    interview_graph = builder.compile(checkpointer=False).with_config(
        run_name="Conduct Interviews"
    )
    final_step = None

    initial_state = {
        "editor": perspectives.editors[0],
        "messages": [
            AIMessage(
                content=f"So you said you were writing an article on {example_topic}?",
                name="Subject_Matter_Expert",
            )
        ],
    }
    async for step in interview_graph.astream(initial_state):
        name = next(iter(step))

    final_step = step

    refine_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Private Equity Investor. You have gathered information from experts and search engines. Now, you are refining the outline of the Investment memo. \
    You need to make sure that the outline is comprehensive and specific. \
    Topic you are writing about: {topic} 

    Old outline:

    {old_outline}""",
            ),
            (
                "user",
                "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Investment memo outline:",
            ),
        ]
    )
    final_state = next(iter(final_step.values()))
    # Using turbo preview since the context can get quite long
    refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
        Outline
    )
    refined_outline = refine_outline_chain.invoke(
        {
            "topic": example_topic,
            "old_outline": initial_outline.as_str,
            "conversations": "\n\n".join(
                f"### {m.name}\n\n{m.content}" for m in final_state["messages"]
            ),
        }
    )
    display_tabs.refined_outline = refined_outline.as_str

    from langchain_community.vectorstores import InMemoryVectorStore
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    reference_docs = [
        Document(page_content=v, metadata={"source": k})
        for k, v in final_state["references"].items()
    ]
    # This really doesn't need to be a vectorstore for this size of data.
    # It could just be a numpy matrix. Or you could store documents
    # across requests if you want.
    vectorstore = InMemoryVectorStore.from_documents(
        reference_docs,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(k=3)


    class SubSection(BaseModel):
        subsection_title: str = Field(..., title="Title of the subsection")
        content: str = Field(
            ...,
            title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
        )

        @property
        def as_str(self) -> str:
            return f"### {self.subsection_title}\n\n{self.content}".strip()

    class WikiSection(BaseModel):
        section_title: str = Field(..., title="Title of the section")
        content: str = Field(..., title="Full content of the section")
        subsections: Optional[List[Subsection]] = Field(
            default=None,
            title="Titles and descriptions for each subsection of the Investment memo.",
        )
        citations: List[str] = Field(default_factory=list)

        @property
        def as_str(self) -> str:
            subsections = "\n\n".join(
                subsection.as_str for subsection in self.subsections or []
            )
            citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
            return (
                    f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
                    + f"\n\n{citations}".strip()
            )

    section_writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Private Equity investor and Investment memo writer. Complete your assigned Investment memo from the following outline:\n\n"
                "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
            ),
            ("user", "Write the full Investment memo section for the {section} section."),
        ]
    )

    async def retrieve(inputs: dict):
        docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
        formatted = "\n".join(
            [
                f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
        return {"docs": formatted, **inputs}

    section_writer = (
            retrieve
            | section_writer_prompt
            | long_context_llm.with_structured_output(WikiSection)
    )
    section = await section_writer.ainvoke(
        {
            "outline": refined_outline.as_str,
            "section": refined_outline.sections[1].section_title,
            "topic": example_topic,
        }
    )


    from langchain_core.output_parsers import StrOutputParser

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Private Equity Investor and an expert Investment memo author. Write the complete Investment memo on an investment in {topic} using the following section drafts:\n\n"
                "{draft}\n\nStrictly follow Investment memo format guidelines.",
            ),
            (
                "user",
                'Write the complete Investment memo using markdown format. Organize citations using footnotes like "[1]",'
                " avoiding duplicates in the footer. Include URLs in the footer.",
            ),
        ]
    )

    writer = writer_prompt | long_context_llm | StrOutputParser()


    class ResearchState(TypedDict):
        topic: str
        outline: Outline
        editors: List[Editor]
        interview_results: List[InterviewState]
        # The final sections output
        sections: List[WikiSection]
        article: str

    async def initialize_research(state: ResearchState):
        topic = state["topic"]
        coros = (
            generate_outline_direct.ainvoke({"topic": topic}),
            survey_subjects.ainvoke(topic),
        )
        results = await asyncio.gather(*coros)
        return {
            **state,
            "outline": results[0],
            "editors": results[1].editors,
        }

    async def conduct_interviews(state: ResearchState):
        topic = state["topic"]
        initial_states = [
            {
                "editor": editor,
                "messages": [
                    AIMessage(
                        content=f"So you said you were writing an article on {topic}?",
                        name="Subject_Matter_Expert",
                    )
                ],
            }
            for editor in state["editors"]
        ]
        # We call in to the sub-graph here to parallelize the interviews
        interview_results = await interview_graph.abatch(initial_states)
        display_tabs.interview_results = interview_results

        return {
            **state,
            "interview_results": interview_results,
        }

    def format_conversation(interview_state):
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
        return f'Conversation with {interview_state["editor"].name}\n\n' + convo

    async def refine_outline(state: ResearchState):
        convos = "\n\n".join(
            [
                format_conversation(interview_state)
                for interview_state in state["interview_results"]
            ]
        )

        updated_outline = await refine_outline_chain.ainvoke(
            {
                "topic": state["topic"],
                "old_outline": state["outline"].as_str,
                "conversations": convos,
            }
        )
        display_tabs.updated_outline = updated_outline
        return {**state, "outline": updated_outline}

    async def index_references(state: ResearchState):
        all_docs = []
        for interview_state in state["interview_results"]:
            reference_docs = [
                Document(page_content=v, metadata={"source": k})
                for k, v in interview_state["references"].items()
            ]
            all_docs.extend(reference_docs)
        await vectorstore.aadd_documents(all_docs)
        return state

    async def write_sections(state: ResearchState):
        outline = state["outline"]
        sections = await section_writer.abatch(
            [
                {
                    "outline": refined_outline.as_str,
                    "section": section.section_title,
                    "topic": state["topic"],
                }
                for section in outline.sections
            ]
        )
        return {
            **state,
            "sections": sections,
        }

    async def write_article(state: ResearchState):
        topic = state["topic"]
        sections = state["sections"]
        draft = "\n\n".join([section.as_str for section in sections])
        article = await writer.ainvoke({"topic": topic, "draft": draft})
        return {
            **state,
            "article": article,
        }

    from langgraph.checkpoint.memory import MemorySaver

    builder_of_storm = StateGraph(ResearchState)

    nodes = [
        ("init_research", initialize_research),
        ("conduct_interviews", conduct_interviews),
        ("refine_outline", refine_outline),
        ("index_references", index_references),
        ("write_sections", write_sections),
        ("write_article", write_article),
    ]
    for i in range(len(nodes)):
        name, node = nodes[i]
        builder_of_storm.add_node(name, node, retry=RetryPolicy(max_attempts=3))
        if i > 0:
            builder_of_storm.add_edge(nodes[i - 1][0], name)

    builder_of_storm.add_edge(START, nodes[0][0])
    builder_of_storm.add_edge(nodes[-1][0], END)
    storm = builder_of_storm.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "my-thread"}}
    async for step in storm.astream(
            {
                "topic": f"A considered, researched and balanced assessment of the risks and upside of a Private Equity Investment in {company}.",
            },
            config,
    ):
        name = next(iter(step))

    checkpoint = storm.get_state(config)
    article = checkpoint.values["article"]
    from IPython.display import Markdown

    # We will down-header the sections to create less confusion in this notebook

    with open("investment_memo.md", "w", encoding="utf-8") as f:
        f.write(article)


    return article, display_tabs


def get_editor_info(editor):
    """Safely extract editor information regardless of type"""
    if isinstance(editor, dict):
        return {
            'name': editor.get('name', ''),
            'role': editor.get('role', ''),
            'affiliation': editor.get('affiliation', ''),
            'description': editor.get('description', '')
        }
    else:
        # Handle Pydantic model
        return {
            'name': getattr(editor, 'name', ''),
            'role': getattr(editor, 'role', ''),
            'affiliation': getattr(editor, 'affiliation', ''),
            'description': getattr(editor, 'description', '')
        }


def get_message_info(message):
    """Safely extract message information regardless of type"""
    if isinstance(message, dict):
        return {
            'content': message.get('content', ''),
            'name': message.get('name', '')
        }
    else:
        return {
            'content': getattr(message, 'content', ''),
            'name': getattr(message, 'name', '')
        }


def get_result_info(result):
    """Safely extract result information regardless of type"""
    if isinstance(result, dict):
        editor = result.get('editor', {})
        messages = result.get('messages', [])
        references = result.get('references', {})
    else:
        editor = getattr(result, 'editor', {})
        messages = getattr(result, 'messages', [])
        references = getattr(result, 'references', {})

    return {
        'editor': get_editor_info(editor),
        'messages': [get_message_info(msg) for msg in messages],
        'references': references
    }


st.set_page_config(page_title="APIs Partners PE Investment Memo App", layout="wide")

# Sidebar
with st.sidebar:
    st.image("apis.png", width=200)  # Adjust path and size
    st.write("Apis Partners")
    st.image("lucidate.png", width=120)  # Another logo
    st.write("Powered by Lucidate")

# Main pane
st.title("APIs Partners PE Investment Memo App")
st.write("Enter company name:")

col1, col2 = st.columns([2,1])
company_name = col1.text_input("", value="")  # empty default
compile_button = col2.button("Compile investment memo")

if compile_button:
    if company_name.strip() == "":
        st.error("Please enter a company name.")
    else:
        with st.spinner(f"Compiling Investment memo for '{company_name}' from primary sources using 'STORM'..."):
            # Run the asynchronous function
            final_memo, display_tabs = asyncio.run(compile_investment_memo(company_name))
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Memo",
                "Outline Evolution",
                "Research Process",
                "Expert Interviews",
                "Conversation Analysis",  # New tab
                "Source Analysis"
            ])

            with tab1:
                col3, col4 = tab1.columns([1, 1])
                col3.markdown(final_memo)

            with tab2:
                # Show how the outline evolved
                st.subheader("Three-stage evolution of the Investment Memo Structure")

                # Use smaller columns to prevent text from being too squeezed
                col_initial, col_refined, col_final = st.columns([1, 1, 1])

                # Add containers for better visual separation
                with col_initial:
                    with st.container():
                        st.markdown("#### 1️⃣ Initial AI-Generated Outline")
                        st.markdown(display_tabs.initial_outline)

                with col_refined:
                    with st.container():
                        st.markdown("#### 2️⃣ Refined Outline After Research")
                        st.markdown(display_tabs.refined_outline)

                with col_final:
                    with st.container():
                        if display_tabs.updated_outline:
                            st.markdown("#### 3️⃣ Final Updated Outline")

                            # Format the final outline properly
                            outline = display_tabs.updated_outline
                            formatted_text = f"# {outline.page_title}\n\n"

                            for section in outline.sections:
                                formatted_text += f"## {section.section_title}\n"
                                formatted_text += f"{section.description}\n\n"

                                if section.subsections:
                                    for subsection in section.subsections:
                                        formatted_text += f"### {subsection.subsection_title}\n"
                                        formatted_text += f"{subsection.description}\n\n"
                                else:
                                    formatted_text += "\n"

                            st.markdown(formatted_text)
            with tab3:
                st.subheader("Research and Analysis Process")

                # Show related subjects identified for research
                st.markdown("#### Related Topics Identified")
                for topic in display_tabs.related_subjects.topics:
                    st.markdown(f"- {topic}")

                # Show search queries generated
                st.markdown("#### Research Queries Generated")
                for query in display_tabs.queries:
                    st.markdown(f"- {query}")

            with tab4:
                st.subheader("Expert Interview Simulations")

                # Debug section
                st.write("DEBUG INFO:")
                st.write("Type of perspectives:", type(display_tabs.perspectives))
                st.write("Type of editors:", type(display_tabs.perspectives['editors']))
                if display_tabs.perspectives['editors']:
                    st.write("Type of first editor:", type(display_tabs.perspectives['editors'][0]))

                st.write("Type of interview_results:", type(display_tabs.interview_results))
                if display_tabs.interview_results:
                    st.write("First interview result:", type(display_tabs.interview_results[0]))
                    st.json(display_tabs.interview_results[0])  # This will show us the structure

                # Then let's try a simplified version of the display logic
                for i, editor in enumerate(display_tabs.perspectives['editors']):
                    with st.expander(f"Expert {i + 1}: Details"):
                        st.write("Editor object type:", type(editor))
                        st.json(editor)  # Show raw editor data

                        if display_tabs.interview_results:
                            st.markdown("#### Conversation:")
                            st.write("Number of interview results:", len(display_tabs.interview_results))
                            for result in display_tabs.interview_results:
                                st.write("Result type:", type(result))
                                st.json(result)  # Show raw result data

            with tab5:
                st.subheader("Conversation Analysis")

                # Debug information
                st.write("Number of editors:", len(display_tabs.perspectives['editors']))
                st.write("Number of interview results:", len(display_tabs.interview_results))

                for i, editor in enumerate(display_tabs.perspectives['editors']):
                    editor_info = get_editor_info(editor)

                    # Debug information for each editor
                    st.write(f"\nLooking for conversations for: {editor_info['name']}")

                    with st.expander(f"Conversation with {editor_info['name']} ({editor_info['role']})"):
                        # Show expert's background first
                        st.markdown("#### Expert Profile")
                        st.markdown(f"""
                        - **Role:** {editor_info['role']}
                        - **Affiliation:** {editor_info['affiliation']}
                        - **Expertise:** {editor_info['description']}
                        """)

                        # Show Q&A flow
                        st.markdown("#### Conversation Flow")

                        # Debug each result match attempt
                        for result in display_tabs.interview_results:
                            result_info = get_result_info(result)
                            st.write(f"Comparing {result_info['editor']['name']} with {editor_info['name']}")

                            if result_info['editor']['name'] == editor_info['name']:

                                # Create a visually appealing Q&A format
                                for msg in result_info['messages']:
                                    # Extract the actual content from AIMessage structure
                                    # Handle both string content and AIMessage objects
                                    if isinstance(msg, str):
                                        content = msg
                                        name = "Unknown"
                                    else:
                                        # Check if the content is nested in an AIMessage structure
                                        if 'AIMessage' in str(msg):
                                            # Extract content from AIMessage structure
                                            import re

                                            content_match = re.search(r"content='(.*?)',", str(msg))
                                            name_match = re.search(r"name='(.*?)'", str(msg))

                                            content = content_match.group(1) if content_match else "No content"
                                            name = name_match.group(1) if name_match else "Unknown"
                                        else:
                                            content = msg.get('content', 'No content')
                                            name = msg.get('name', 'Unknown')

                                    # Style based on who's speaking
                                    if name == 'Subject_Matter_Expert':
                                        st.markdown(f"""
                                        <div style='background-color: #000206; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <strong>Expert:</strong> {content}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style='background-color: #182429; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <strong>{name}:</strong> {content}
                                        </div>
                                        """, unsafe_allow_html=True)

        # Once done, display the final memo
        # st.markdown(final_memo)
