import modal

app = modal.App("tweet-vectors-test")

@app.function()
def hello():
    print("Hello from Modal!")
    return "Modal is working! Ready to deploy Milvus and Voyage AI embeddings."

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(result)
