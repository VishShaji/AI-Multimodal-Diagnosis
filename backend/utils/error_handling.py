class ModelNotFoundError(Exception):
    pass

class InvalidInputError(Exception):
    pass

def handle_model_not_found(model_name):
    raise ModelNotFoundError(f"Model '{model_name}' not found.")

def handle_invalid_input(input_data, expected_type):
    raise InvalidInputError(f"Invalid input. Expected {expected_type}, got {type(input_data)}.")