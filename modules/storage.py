from sentence_transformers import SentenceTransformer
from modules.utilis.utilities import *


class Storage:

    def __init__(self, storage_path, data_path, type_encoding, name_storage='storage'):
        self.storage_path = storage_path
        self.data = load_data(data_path)
        self.type_encoding = type_encoding
        self.storage_file_path = os.path.join(storage_path, f'{name_storage}.pt')

        # Check if the storage  exists
        if not os.path.exists(self.storage_file_path):
            print(f"Creating directory for {name_storage}.pt ...")
            create_dir(storage_path)
            print("Populating storage with embeddings...")
            self.populate_storage()
        else:
            print("Storage directory already exists.")
        print('-' * 50)

    def get_storage_data(self):
        return self.data

    def get_embedding(self):
        return torch.load(self.storage_file_path)

    def populate_storage(self):
        model = SentenceTransformer('all-mpnet-base-v2')
        questions = [create_input_to_embedd(question_dict, self.type_encoding) for question_dict in self.data.values()]
        embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
        save_tensor(self.storage_file_path, embeddings)
