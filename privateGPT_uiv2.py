import gradio as gr
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma
import os
import argparse

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

def clear_history(request: gr.Request):
    state = None
    return ([], state, "")

def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code

def post_process_answer(answer, source):
    answer += f"<br><br>Source: {source}"
    answer = answer.replace("\n", "<br>")
    return answer

def predict(
    question: str,
    system_content: str,
    embeddings: str,
    persist_directory: str,
    model_type: str,
    model_path: str,
    model_n_ctx: int,
    chatbot: list = [],
    history: list = [],
):
    try:
        # Parse the command line arguments      
        args = parse_arguments()

        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()
        
        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
        
        # Prepare the LLM
        if model_type == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_threads=16)
        elif model_type == "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False, n_threads=16)
        else:
            print(f"Model {model_type} not supported!")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
        
        # Get the answer from the chain
        prompt = system_content + f"\n Question: {question}"
        res = qa(prompt)    
        answer, docs = res['result'], res['source_documents']
        answer = post_process_answer(answer, docs)
        history.append(question)
        history.append(answer)
        chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
        return chatbot, history
    
    except Exception as e:
        history.append("")
        answer = server_error_msg + f" (error_code: 503)"
        history.append(answer)
        chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
        return chatbot, history

def reset_textbox():
    return gr.update(value="")

def main():
    title = """<h1 align="center">Chat with privateGPT 🤖</h1>"""

    with gr.Blocks(
        css="""
        footer .svelte-1lyswbr {display: none !important;}
        #col_container {margin-left: auto; margin-right: auto;}
        #chatbot .wrap.svelte-13f7djk {height: 70vh; max-height: 70vh}
        #chatbot .message.user.svelte-13f7djk.svelte-13f7djk {width:fit-content; background:orange; border-bottom-right-radius:0}
        #chatbot .message.bot.svelte-13f7djk.svelte-13f7djk {width:fit-content; padding-left: 16px; border-bottom-left-radius:0}
        #chatbot .pre {border:2px solid white;}
        pre {
        white-space: pre-wrap;       /* Since CSS 2.1 */
        white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
        white-space: -pre-wrap;      /* Opera 4-6 */
        white-space: -o-pre-wrap;    /* Opera 7 */
        word-wrap: break-word;       /* Internet Explorer 5.5+ */
        }
        """
    ) as demo:
        gr.HTML(title)
        with gr.Row():
            with gr.Column(elem_id="col_container", scale=0.3):
                with gr.Accordion("Prompt", open=True):
                    system_content = gr.Textbox(value="You are headcountGPT, built with LangChain and GPT4All. Answer questions based on the documents you have ingested. Answer 'I don't know' if you can't find the answer in the document.", show_label=False)
                with gr.Accordion("Config", open=True):
                    embeddings_model_name = gr.Textbox(value="models/ggml-model-q4_0.bin", label="embeddings_model_name")
                    persist_directory = gr.Textbox(value="db", label="persist_directory")
                    model_type = gr.Textbox(value="GPT4All", label="model_type")
                    model_path = gr.Textbox(value="models/ggml-gpt4all-j-v1.3-groovy.bin", label="model_path")
                    model_n_ctx = gr.Slider(
                        minimum=32,
                        maximum=4096,
                        value=1000,
                        step=32,
                        interactive=True,
                        label="model_n_ctx",
                    )
                    
            with gr.Column(elem_id="col_container"):
                chatbot = gr.Chatbot(elem_id="chatbot", label="privateGPT")
                question = gr.Textbox(placeholder="Ask something", show_label=False, value="")
                state = gr.State([])
                with gr.Row():
                    with gr.Column():
                        submit_btn = gr.Button(value="🚀 Send")
                    with gr.Column():
                        clear_btn = gr.Button(value="🗑️ Clear history")
                    
        question.submit(
            predict,
            [question, system_content, embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, chatbot, state],
            [chatbot, state],
        )
        submit_btn.click(
            predict,
            [question, system_content, embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, chatbot, state],
            [chatbot, state],
        )
        submit_btn.click(reset_textbox, [], [question])
        clear_btn.click(clear_history, None, [chatbot, state, question])
        question.submit(reset_textbox, [], [question])
        demo.queue(concurrency_count=10, status_update_rate="auto")
        #demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share, debug=args.debug)
        demo.launch()

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == '__main__':
    """ import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", default=8071)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args() """
    main()