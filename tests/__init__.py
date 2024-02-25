import os

# Check if the environment variable PEFT_DEBUG_WITH_TORCH_COMPILE is set to 1
if os.environ.get("PEFT_DEBUG_WITH_TORCH_COMPILE") == "1":
    # If it is, this is a debugging hack to return a compiled model from get_peft_model()
    # This allows all unit tests that use peft.get_peft_model() to use a compiled model

    # Import necessary modules
    import torch
    import peft
    from peft.mapping import get_peft_model as get_peft_model_original

    # Define a new version of get_peft_model() that returns a compiled model
    def get_peft_model_new(*args, **kwargs):
        """Make get_peft_model() return a compiled model."""
        # Call the original get_peft_model() to get the uncompiled model
        peft_model = get_peft_model_original(*args, **kwargs)
        # Compile the model using torch.compile()
        peft_model = torch.compile(peft_model)
        # Return the compiled model
        return peft_model

    # Replace the original get_peft_model() with the new version
    peft.get_peft_model = get_peft_model_new
