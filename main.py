import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Building_blocks import laplacian_decompose, LowFrequencyModule, HighFrequencyModule
from Dataset_preparation import SD7KDataset
from train import train_hybrid_model, save_generated_image
from Hybrid_model import HybridShadowRemoval

"""
A architecture which is combination of Document shadow removal (FSENet) and 
shadow removal architecture (LSRNet and DRNet).
"""


def main(num_epochs=10, batch_size=2, lr=0.0002):
    # Path to the SD7K dataset root folder.
    dataset_root = "SD7K"  # Replace with the correct path.
    # For demonstration, we use a target resolution of 1024x1024.
    target_res = (1024, 1024)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)

    # Ensure the directory to save the model exists.
    os.makedirs("saved_models", exist_ok=True)
    hybrid_net_path = os.path.join("saved_models", "hybrid_shadow_removal.pth")

    # Create dataset and dataloader for training.
    train_dataset = SD7KDataset(dataset_root, phase='train', target_res=target_res)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create dataset and dataloader for testing.
    test_dataset = SD7KDataset(dataset_root, phase='test', target_res=target_res)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize the hybrid model.
    model = HybridShadowRemoval().to(device)

    # If more than one GPU is available, use DataParallel.
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "cuda GPUs")
        model = nn.DataParallel(model)

    # Define loss function and optimizer.
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(hybrid_net_path):
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(hybrid_net_path, map_location=device))
        print(f"Loaded saved Hybrid model weights from {hybrid_net_path}")
    else:
        # Training loop.
        for epoch in range(num_epochs):
            epoch_loss = train_hybrid_model(model, train_loader, optimizer, criterion, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        # Save the trained model weights.
        torch.save(model.state_dict(), hybrid_net_path)
        print(f"Model weights saved to: {hybrid_net_path}")

    # Generate and save a few test images.
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inp = batch['input'].to(device)
            output = model(inp).cpu()
            filename = batch['filenames'][0]
            save_generated_image(output[0], "generated_outputs", filename)
            if 40 <= idx <= 50:
                break  # Save 10 samples.


if __name__ == "__main__":
    main(num_epochs=70, batch_size=4, lr=0.0002)
