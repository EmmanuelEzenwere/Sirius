import pickle

def load_model(model_path):
    """
    Load a machine learning model from the specified path.

    Args:
        model_path (str): The file path to the saved model.

    Returns:
        The loaded machine learning model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return model




