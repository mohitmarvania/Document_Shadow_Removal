import os
from torchvision.utils import save_image


# Training function
def train_hybrid_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['input'].to(device)  # Use high-res image as input.
        gt = batch['target'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def save_generated_image(tensor, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    save_image(tensor, save_path)
    print(f"Saved generated image: {save_path}")
