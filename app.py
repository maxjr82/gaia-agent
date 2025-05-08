# app.py

import gradio as gr
from typing import List
from langchain_core.messages import HumanMessage
from agent_v2 import gaia_agent

# A small list of models to choose from
MODEL_OPTIONS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.5-pro-exp-03-25",
    "openai/gpt-4o-mini",
]

# A few sample GAIA questions
EXAMPLES = [
    "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?",
    "If a person is 12 years younger than the French president, Emmanuel Macron, what age is the person?",
    "In NASA’s Astronomy Picture of the Day on 2006 January 21, which astronaut from the smaller appearance spent the least time in space, and how many minutes did he spend? Provide last name and minutes.",
]


def _create_agent(model_name: str):
    """Instantiate (or re‐instantiate) the GAIA agent with the selected model."""
    return gaia_agent(model=model_name)


def chat_with_agent(history: List[List[str]], user_input: str, model_name: str):
    """
    Receives the current chat history and a new user message,
    returns updated history.
    """
    # 1. Initialize history if empty
    if history is None:
        history = []

    # 2. Append the user's turn
    history.append(("User", user_input))

    # 3. (Re)create the agent and invoke it
    agent = _create_agent(model_name)
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    # 4. Collect all assistant messages into one block
    if isinstance(response, dict) and "messages" in response:
        # Join every assistant message in order
        assistant_contents = [msg.content for msg in response["messages"]]
        reply = "\n\n".join(assistant_contents).strip()
    else:
        reply = str(response).strip()

    # 5. Append the full assistant reply
    history.append(("Agent", reply))
    return history


with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 🤖 GAIA Agent Chat Interface")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Select model",
            choices=MODEL_OPTIONS,
            value=MODEL_OPTIONS[0],
            interactive=True,
        )
        example_buttons = gr.Dropdown(
            label="Or pick an example question",
            choices=EXAMPLES,
            value=None,
            interactive=True,
        )
    chatbot = gr.Chatbot(label="GAIA Agent", height=500)
    user_input = gr.Textbox(
        placeholder="Type your GAIA question here...",
        label="Your question",
        lines=2,
    )
    send_button = gr.Button("Send")

    # When an example is chosen, fill the input box
    example_buttons.change(fn=lambda q: q, inputs=example_buttons, outputs=user_input)

    # Send on button click or enter
    send_button.click(
        fn=chat_with_agent,
        inputs=[chatbot, user_input, model_dropdown],
        outputs=chatbot,
    )
    user_input.submit(
        fn=chat_with_agent,
        inputs=[chatbot, user_input, model_dropdown],
        outputs=chatbot,
    )

    gr.Markdown(
        """
        **Notes**  
        - The agent is re-instantiated when you change the model.  
        - You can copy/paste additional GAIA questions or pick from the examples.  
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
