import streamlit as st
import re
import pickle


import os

from dotenv import load_dotenv
import getpass

import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

EXPERT_NAMES = [
    "Alice Anderson",
    "Bob Bennett",
    "Charlie Cooper",
    "David Davidson",
    "Emma Edwards",
    "Frank Fletcher",
    "Grace Gardner",
    "Henry Harrison",
    "Iris Irving",
    "James Johnson"
]

def get_expert_name(index: int) -> str:
    """Get a consistent expert name from our predefined pool"""
    return EXPERT_NAMES[index % len(EXPERT_NAMES)]

def save_display_tabs(display_tabs, company_name):
    with open(f'display_tabs_{company_name}.pkl', 'wb') as f:
        pickle.dump(display_tabs, f)
def _set_env(var: str):
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var + ":")

def escape_dollar_signs(text: str) -> str:
    """Replace $ with \$ to prevent markdown interpretation"""
    return text.replace('$', r'\$')


def state_graph_to_mermaid(graph, title="Workflow") -> str:
    """Convert a LangGraph StateGraph to Mermaid diagram markup"""
    # Get the underlying NetworkX graph with execution info
    g = graph.get_state(xray=True)  # This is how we access the graph from langgraph

    # Start the Mermaid diagram
    mermaid_code = [
        "%%{init: {'theme': 'dark'}}%%",
        "graph TD",
        f"    title[{title}]"
    ]

    # Add nodes and edges
    seen_nodes = set()

    # Helper function to sanitize node names for Mermaid
    def sanitize_node(node):
        if node == "START":
            return "((START))"
        elif node == "END":
            return "((END))"
        else:
            return f"[{node}]"

    # Extract edges from the graph
    for node, edges in g.config.nodes.items():
        if node not in seen_nodes:
            seen_nodes.add(node)

        for edge in edges:
            target = edge.target
            if target not in seen_nodes:
                seen_nodes.add(target)

            # Add the edge to Mermaid markup
            mermaid_code.append(f"    {node}{sanitize_node(node)} --> {target}{sanitize_node(target)}")

    # Add styling
    mermaid_code.extend([
        "    classDef default fill:#1E88E5,stroke:#4A90E2,color:white",
        "    classDef start fill:#4CAF50,stroke:#45A049,color:white",
        "    classDef end fill:#F44336,stroke:#D32F2F,color:white",
        "    class START start",
        "    class END end"
    ])

    return "\n".join(mermaid_code)


# Then in your tab7:



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
_set_env("OPENAI_API_KEY")
# _set_env("TAVILY_API_KEY")


def sanitize_name(name: str) -> str:
    """
    Consistently sanitize names while preserving readability.
    Converts "Alice Anderson" to "Alice_Anderson"
    """
    # Replace spaces with underscores, remove any other special characters
    sanitized = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return sanitized.replace(' ', '_')

async def compile_investment_memo(company):
    from langchain_openai import ChatOpenAI

    fast_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    # Uncomment for a Fireworks model
    # fast_llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v1", max_tokens=32_000)
    long_context_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    from typing import List, Optional, Any

    from langchain_core.prompts import ChatPromptTemplate

    from pydantic import BaseModel, Field, validator

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
            self.storm_graph: Any
            self.interview_graph: Any


    display_tabs = IntermediateResults()

    direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a private equity analyst. Create a comprehensive due diligence outline about a user-provided "
                "company or investment opportunity. Include sections on market analysis, competitive landscape, "
                "financial performance, detailed SWOT analysis, risks, any prior investors, valuations and investment "
                "rounds as well as potential returns.",
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

    example_topic = f"Evaluating a potential private equity investment in {company}, a startup FinTech firm."

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
            pattern=r"^[A-Za-z ]{1,64}$"  # Only letters and spaces
        )
        role: str = Field(
            description="Role of the editor in the context of the topic.",
        )
        description: str = Field(
            description="Description of the editor's focus, concerns, and motives.",
        )

        @validator('name')
        def validate_name(cls, name):
            """Ensure name is from our predefined list"""
            if name not in EXPERT_NAMES:
                raise ValueError(f"Name must be one of the predefined expert names: {', '.join(EXPERT_NAMES)}")
            return name

        @property
        def persona(self) -> str:
            return f"Name: {sanitize_name(self.name)}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    class Perspectives(BaseModel):
        editors: List[Editor] = Field(
            description="Comprehensive list of editors with their roles and affiliations.",
            # Add a pydantic validation/restriction to be at most M editors
        )

    gen_perspectives_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You need to select a diverse group of Investment analysts who will work together to create a comprehensive Investment memo on the topic. 
            Please use ONLY the following names for the analysts (in order): {expert_names}

            Each analyst represents a different perspective, role, or affiliation related to this company, investment opportunity and topic.
            You can use other Investment opportunity pages of related topics for inspiration. For each editor, add a description of what they will focus on.

            Investment memo outlines of related topics for inspiration:
            {examples}""",
        ),
        ("user", "Topic of interest: {topic}"),
    ])

    gen_perspectives_chain = gen_perspectives_prompt | ChatOpenAI(
        model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
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
        return await gen_perspectives_chain.ainvoke({
                "examples": formatted,
                "topic": topic,
                "expert_names": ", ".join(EXPERT_NAMES)
            })

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

    gen_qn_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an experienced Private Equity Investment manager conducting due diligence on a potential investment.
            You have a very specific area of expertise which is based on the Description: in {description}. You must maintain focus on this specific area in your questioning:

            You are chatting with an industry expert to gather information that relates SPECIFICALLY to your area of expertise, which is based on your Role: and Affiliation: from {persona}.
            Your questions should draw directly from your role, affiliation and expertise description.

            For example:
            - A Legal Expert should focus on regulatory compliance, intellectual property, contractual obligations
            - A Data Analyst should focus on user metrics, engagement patterns, performance indicators
            - A Tech Expert should focus on system architecture, scalability, technical debt
            - A Financial Expert should focus on revenue models, cost structures, margins
            - A Market Analyst should focus on competition, competitive threats, supplier and customer dynamics 
            - An Investment Analyst should focus on existing investors and valuations in prior investment rounds in this company or companies in this sector

            Ask ONE question at a time about the target company, but ensure each question is directly related to your specific expertise; i.e.  {description}.
            Do not ask about general topics outside your domain of expertise.
            Do not ask questions that other experts would be better suited to ask.

            When you have no more questions specific to your domain of expertise, say "Thank you so much for your help!"
            """
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ])


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
                | gen_qn_prompt.partial(persona=editor.persona, description=editor.description)
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
        model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
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

    from langchain_community.tools.tavily_search import TavilySearchResults
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
        try:
            # Try Tavily first
            # tavily_search = TavilySearchResults(
            #     max_results=4,
            #     api_key=TAVILY_API_KEY  # We already have this from environment
            # )
            # results = tavily_search.invoke(query)
            results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
            print(f"DDG Results: {results}")
            return [{"content": r["content"], "url": r["url"]} for r in results]
        except Exception as e:
            print(f"Tavily search failed: {str(e)}. Falling back to DuckDuckGo.")
            # Fall back to DuckDuckGo
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
    display_tabs.interview_graph = interview_graph  # Store the graph
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
    display_tabs.storm_graph = storm  # Store the graph

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

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# The same fallback search logic as your code
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# 1) A function to run the actual web search
def do_web_search(query: str, max_results=4):
    """Tries Tavily first, then DuckDuckGo."""
    # tavily = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=max_results)
    duck = DuckDuckGoSearchAPIWrapper()
    try:
        results = None
        if results:
            return results
        print("No results from Tavily. Falling back to DuckDuckGo.")
        ddg_results = duck._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in ddg_results]
    except Exception as e:
        print(f"Tavily search failed: {e}. Falling back to DuckDuckGo.")
        ddg_results = duck._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in ddg_results]

# 2) A function to extract company names from the search result text
def extract_company_names(user_request: str, web_results) -> list:
    """
    Pass the web results to an LLM prompt that says:
      "Based on the user request, list possible relevant company names only."
    Return a Python list of strings.
    """
    text_snippets = []
    for r in web_results:
        snippet = (r["content"] or "")[:800]  # limit text
        text_snippets.append(f"- {snippet}")

    combined_text = "\n".join(text_snippets)

    prompt = ChatPromptTemplate.from_template(
        """
        The user asked: {user_request}

        Here are some search snippets that might mention relevant companies:
        {combined_text}

        Please identify up to 10 distinct company names that match the user's request.
        Return them as a JSON list of strings, e.g. ["CompanyA", "CompanyB"].
        """
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    resp = llm.invoke(
        prompt.format(user_request=user_request, combined_text=combined_text)
    )

    # Attempt to parse JSON
    try:
        data = json.loads(resp.content)
        # ensure it's a list of strings
        if isinstance(data, list):
            # e.g. ["CompanyA", "CompanyB"]
            return [str(x) for x in data]
        else:
            return []
    except:
        return []

def find_companies(user_request: str) -> list:
    """High-level function: search, then extract relevant company names."""
    raw_results = do_web_search(user_request, max_results=4)
    companies = extract_company_names(user_request, raw_results)
    return companies






st.set_page_config(page_title="NEXTfrontier Investment Memo App", layout="wide")

if "found_companies" not in st.session_state:
    st.session_state["found_companies"] = []
if "selected_company" not in st.session_state:
    st.session_state["selected_company"] = None

# Sidebar
with st.sidebar:
    st.image("lucidate.png", width=120)  # Another logo
    st.write("Powered by Lucidate")
    st.write("## Find Matching Companies")
    search_text = st.text_area("Describe the companies you want to find:")
    max_dropdown = st.number_input("Max companies to show:", min_value=0, max_value=10, value=3)

    if st.button("Search for Companies"):
        st.session_state["found_companies"] = find_companies(search_text)
        st.write("Companies found:")
        for c in st.session_state["found_companies"]:
            st.write("-", c)

# Main pane
st.image("nfc.png", width=200)  # Adjust path and size
st.write("NEXTfrontier")
st.title("NEXTfrontier Capital Investment Memorandum Writer")
st.write("Enter company name:")

# Let user pick a company from the found list
if st.session_state["found_companies"]:
    st.session_state["selected_company"] = st.selectbox(
        "Select a company to build a memo for:",
        st.session_state["found_companies"][:max_dropdown],  # limit how many to show
    )

compile_button = st.button("Build Memo", disabled=not st.session_state["selected_company"])


if compile_button and st.session_state["selected_company"]:
    company_name = st.session_state["selected_company"]
    with st.spinner(f"Compiling Investment memo for '{company_name}' from primary sources using 'STORM'..."):
        final_memo, display_tabs = asyncio.run(compile_investment_memo(company_name))

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Memo",
            "Outline Evolution",
            "Research Process",
            "Expert Interviews",
            "Conversation Analysis",  # New tab
            "Source Analysis",
            "Process Visualization"
        ])

        with tab1:
            col3, col4 = tab1.columns([1, 1])
            col3.markdown(escape_dollar_signs(final_memo))

        with tab2:
            # Show how the outline evolved
            st.subheader("Three-stage evolution of the Investment Memo Structure")

            # Use smaller columns to prevent text from being too squeezed
            col_initial, col_refined, col_final = st.columns([1, 1, 1])

            # Add containers for better visual separation
            with col_initial:
                with st.container():
                    st.markdown("#### 1️⃣ Initial AI-Generated Outline")
                    st.markdown(escape_dollar_signs(display_tabs.initial_outline))

            with col_refined:
                with st.container():
                    st.markdown("#### 2️⃣ Refined Outline After Research")
                    st.markdown(escape_dollar_signs(display_tabs.refined_outline))

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

                        st.markdown(escape_dollar_signs(formatted_text))
        with tab3:
            st.subheader("Research and Analysis Process")

            # Show related subjects identified for research
            st.markdown("#### Related Topics Identified")
            for topic in display_tabs.related_subjects.topics:
                st.markdown(escape_dollar_signs(f"- {topic}"))

            # Show search queries generated
            st.markdown("#### Research Queries Generated")
            for query in display_tabs.queries:
                st.markdown(escape_dollar_signs(f"- {query}"))

        with tab4:
            st.subheader("Expert Interview Simulations")

            # Debug section
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
            st.subheader("Detailed Conversation Analysis")

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
                    st.markdown(escape_dollar_signs(f"""
                    - **Role:** {editor_info['role']}
                    - **Affiliation:** {editor_info['affiliation']}
                    - **Expertise:** {editor_info['description']}
                    """))

                    # Show Q&A flow
                    st.markdown("#### Conversation Flow")

                    # Debug each result match attempt
                    for result in display_tabs.interview_results:
                        result_info = get_result_info(result)
                        # st.write(f"Comparing {result_info['editor']['name']} with {editor_info['name']}")

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
                                        <strong>Expert:</strong> {escape_dollar_signs(content)}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style='background-color: #182429; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                        <strong>{name}:</strong> {escape_dollar_signs(content)}
                                    </div>
                                    """, unsafe_allow_html=True)
        with tab6:
            st.subheader("Source Analysis")

            # Get unique citations from display_tabs
            with st.expander("Referenced URLs"):
                if display_tabs.cited_urls:
                    for i, url in enumerate(display_tabs.cited_urls, 1):
                        st.markdown(escape_dollar_signs(f"{i}. [{url}]({url})"))
                else:
                    st.info("No URLs cited")

            # Show references with their content
            with st.expander("Source Content Analysis"):
                if display_tabs.cited_references:
                    for url, content in display_tabs.cited_references.items():
                        st.markdown(f"### Source: [{url}]({url})")
                        with st.container():
                            st.markdown(f"```\n{content[:500]}...\n```")
                        st.markdown("---")
                else:
                    st.info("No reference content available")

            # Show how sources were used
            with st.expander("Source Usage by Expert"):
                for i, editor in enumerate(display_tabs.perspectives['editors']):
                    editor_info = get_editor_info(editor)
                    st.markdown(f"### {editor_info['name']}'s Sources")

                    expert_citations = set()
                    for result in display_tabs.interview_results:
                        result_info = get_result_info(result)
                        if result_info['editor']['name'] == editor_info['name']:
                            for msg in result_info['messages']:
                                if isinstance(msg, dict) and 'content' in msg:
                                    # Look for citation patterns [1]: url or similar
                                    citations = re.findall(r'\[[\d\^]+\]:\s*(http[s]?://\S+)', msg['content'])
                                    expert_citations.update(citations)

                    if expert_citations:
                        for url in expert_citations:
                            st.markdown(f"- [{url}]({url})")
                    else:
                        st.info(f"No sources cited by {editor_info['name']}")

            # Statistics about sources
            st.markdown("### Source Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Unique Sources",
                          len(display_tabs.cited_urls) if display_tabs.cited_urls else 0)

            with col2:
                # Calculate number of experts citing sources using helper functions
                experts_with_citations = 0
                for result in display_tabs.interview_results:
                    result_info = get_result_info(result)
                    has_citations = False
                    for msg in result_info['messages']:
                        content = msg['content'] if isinstance(msg, dict) else getattr(msg, 'content', '')
                        if '[' in str(content):
                            has_citations = True
                            break
                    if has_citations:
                        experts_with_citations += 1
                st.metric("Experts Citing Sources", experts_with_citations)

            with col3:
                # Average citations per response
                total_citations = 0
                for result in display_tabs.interview_results:
                    result_info = get_result_info(result)
                    for msg in result_info['messages']:
                        content = msg['content'] if isinstance(msg, dict) else getattr(msg, 'content', '')
                        citations = len(re.findall(r'\[[\d\^]+\]', str(content)))
                        total_citations += citations

                avg_citations = total_citations / len(
                    display_tabs.interview_results) if display_tabs.interview_results else 0
                st.metric("Average Citations per Expert", f"{avg_citations:.1f}")

        # ... other tabs ...

        with tab7:
            st.subheader("Process Visualization")

            # First visualization: Agent Network
            st.markdown("### Agent Collaboration Network")

            import networkx as nx
            import matplotlib.pyplot as plt

            # Set dark background style for matplotlib
            plt.style.use('dark_background')

            # Create graph
            G = nx.DiGraph()
            agents = ["Research Coordinator", "Outline Generator", "Expert Interviewer",
                      "Source Analyzer", "Content Writer", "Fact Checker"]

            for agent in agents:
                G.add_node(agent)

            edges = [
                ("Research Coordinator", "Outline Generator"),
                ("Outline Generator", "Expert Interviewer"),
                ("Expert Interviewer", "Source Analyzer"),
                ("Source Analyzer", "Content Writer"),
                ("Content Writer", "Fact Checker"),
                ("Fact Checker", "Research Coordinator")
            ]
            G.add_edges_from(edges)

            # Create visualization with dark theme
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0E1117')
            ax.set_facecolor('#0E1117')

            pos = nx.spring_layout(G)
            nx.draw(G, pos,
                    with_labels=True,
                    node_color='#1E88E5',  # Blue nodes
                    node_size=5000,
                    font_size=6,
                    font_weight='bold',
                    font_color='white',
                    edge_color='#4A90E2',
                    arrows=True,
                    arrowsize=20,
                    ax=ax)

            st.pyplot(fig)



            if display_tabs.storm_graph:
                st.markdown("### STORM Process Flow")
                storm_mermaid = display_tabs.storm_graph.get_graph(xray=True).draw_mermaid_png()
                st.image(storm_mermaid)

            if display_tabs.interview_graph:
                st.markdown("### Interview Process Flow")
                interview_mermaid = display_tabs.interview_graph.get_graph(xray=True).draw_mermaid_png()
                st.image(interview_mermaid)

    # Once done, display the final memo
    # st.markdown(final_memo)
