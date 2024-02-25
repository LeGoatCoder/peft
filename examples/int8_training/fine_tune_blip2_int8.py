# Copyright notice and license information
import torch
from datasets import load_dataset  # For loading datasets
from torch.utils.data import DataLoader, Dataset  # For creating data loaders
from transformers import AutoModelForVision2Seq, AutoProcessor  # For loading pre-trained models and processors

# Import the LoRaConfig and get_peft_model functions from the peft module
from peft import LoraConfig, get_peft_model

# Define the LoRaConfig with specific hyperparameters
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# Load the pre-trained model and processor using the transformers library
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Get the peft model and print the number of trainable parameters
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Load the dataset
dataset = load_dataset("ybelkada/football-dataset", split="train")

# Define the custom dataset class for image captioning
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

# Define the collator function for padding input_ids and attention_mask
def collator(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

# Initialize the train dataset and dataloader
train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Set the device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the model to training mode
model.train()

# Train the model for 50 epochs
for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        # Move input_ids and pixel_values to the device
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        # Forward pass
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

        # Calculate the loss
        loss = outputs.loss

        # Print the loss
        print("Loss:", loss.item())

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

        # Generate and print the output every 10 batches
        if idx % 10 == 0:
            generated_output = model.generate(pixel_values=pixel_values)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))
