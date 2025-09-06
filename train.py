from ultralytics import YOLO

def main():
    # Load medium-sized YOLOv8 model for better performance
    model = YOLO('yolov8n.pt')  # or 'yolov8m.yaml' if training from scratch

    # Train the model
    model.train(
        data='data.yaml',    # Your dataset config
        epochs=100,                  # You can increase up to 150 if GPU allows
        imgsz=640,                   # Image size
        batch=16,                    # Adjust based on your GPU VRAM (try 8 if OOM)
        patience=20,                 # Early stopping patience
        lr0=0.001,                   # Initial learning rate
        lrf=0.01,                    # Final learning rate fraction
        momentum=0.937,              # Momentum
        weight_decay=0.0005,         # Regularization
        warmup_epochs=3,            # Warm-up to stabilize training
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        degrees=0.5,                 # Small rotation
        translate=0.1,               # Small shift
        scale=0.5,                   # Random scale
        shear=0.1,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,                  # Horizontal flip
        mosaic=1.0,                  # Enable mosaic augmentation
        mixup=0.1,
    
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        cache=False,                # Set True if you have enough RAM
        device=0,                   # GPU index
        workers=8,                  # Number of dataloader workers
        optimizer='SGD',            # Better generalization (can try 'Adam')
        pretrained=True,           # Transfer learning if using yolov8m.pt
        val=True,                   # Run validation after training
        project='smart_traffic',    # Output folder
        name='emergency_vehicle_model25', # Model name
        exist_ok=True
    )

if __name__ == '__main__':
    main()
