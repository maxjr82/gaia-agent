# app.py

import gradio as gr
from typing import List, Tuple
from langchain_core.messages import HumanMessage
from agent_v2 import gaia_agent

MODEL_OPTIONS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.5-pro-exp-03-25",
    "openai/gpt-4o-mini",
]

EXAMPLES = [
    "What was the actual enrollment count of the clinical trial on H. pylori ...?",
    "If a person is 12 years younger than the French president, ...?",
    "In NASA’s Astronomy Picture of the Day on 2006 January 21, ...?",
]

def _create_agent(model_name: str):
    return gaia_agent(model=model_name)

def chat_with_agent(
    history: List[Tuple[str,str]],
    user_input: str,
    model_name: str
) -> Tuple[List[Tuple[str,str]], str]:
    if history is None:
        history = []
    history.append(("User", user_input))

    agent = _create_agent(model_name)
    response = agent.invoke({"messages":[HumanMessage(content=user_input)]})
    # concatenate all assistant messages
    assistant_contents = [m.content for m in response.get("messages", [])]
    full_reply = "\n\n".join(assistant_contents).strip()

    history.append(("Agent", full_reply))
    return history, full_reply

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 GAIA Agent Chat Interface")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            "Select model",
            choices=MODEL_OPTIONS,
            value=MODEL_OPTIONS[0],
            interactive=True
        )
        example_dropdown = gr.Dropdown(
            "Or pick an example question",
            choices=EXAMPLES,
            value=None,
            interactive=True
        )

    chatbot = gr.Chatbot(label="Conversation", height=400)
    output_box = gr.Textbox(
        label="Full Agent Response",
        interactive=False,
        lines=10
    )
    user_input = gr.Textbox(
        placeholder="Type your GAIA question here...",
        label="Your Question",
        lines=2
    )
    send = gr.Button("Send")

    # fill input from example
    example_dropdown.change(lambda q: q, inputs=example_dropdown, outputs=user_input)

    # on send: update both chat history and the full-output box
    send.click(
        chat_with_agent,
        inputs=[chatbot, user_input, model_dropdown],
        outputs=[chatbot, output_box]
    )
    user_input.submit(
        chat_with_agent,
        inputs=[chatbot, user_input, model_dropdown],
        outputs=[chatbot, output_box]
    )

    gr.Markdown(
        """
        **Notes**  
        - Changing the model re-instantiates the agent.  
        - The “Full Agent Response” box always displays the complete output.
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
