class LMStudioEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Assuming your LMStudio model has an 'embed' method
        return [self.model.embed(text) for text in texts]

    def embed_query(self, text):
        return self.model.embed(text)

