"""CIFAR dataset loading utilities."""

import pickle
import numpy as np
import jax.numpy as jnp


def unpickle(file):
    """Unpickle CIFAR-10 data files"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(data_dir="../Datasets/cifar-10-batches-py"):
    """Load complete CIFAR-10 dataset"""
    # Training data - 5 batches
    x_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        x_train_list.append(batch[b'data'])
        y_train_list.append(batch[b'labels'])
    
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Test data
    test_batch = unpickle(f"{data_dir}/test_batch")
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # Data preprocessing: reshape and normalize
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_train = x_train.reshape(-1, 3072).astype('float32') / 255.0
    
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3072).astype('float32') / 255.0
    
    # Improved preprocessing: standardize data
    # Compute mean and std for each feature
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8  # Avoid division by zero
    
    # Apply standardization (zero mean, unit variance)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # Convert to JAX arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    
    print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    
    return x_train, y_train, x_test, y_test


def load_tiny_imagenet(data_dir="../Datasets/tiny-imagenet/tiny-imagenet-200"):
    """Load complete Tiny ImageNet dataset"""
    import os
    from PIL import Image
    
    # Load class names and create mapping
    with open(f"{data_dir}/wnids.txt", 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Load training data
    x_train_list = []
    y_train_list = []
    
    print("Loading training data...")
    for class_name in class_names:
        class_dir = f"{data_dir}/train/{class_name}/images"
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img).flatten()  # Flatten to 1D
                        x_train_list.append(img_array)
                        y_train_list.append(class_to_idx[class_name])
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    x_train = np.array(x_train_list, dtype='float32') / 255.0
    y_train = np.array(y_train_list)
    
    # Load validation data (used as test set)
    x_test_list = []
    y_test_list = []
    
    print("Loading validation data...")
    val_annotations = {}
    with open(f"{data_dir}/val/val_annotations.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_name = parts[1]
                val_annotations[img_name] = class_name
    
    val_dir = f"{data_dir}/val/images"
    for img_file in sorted(os.listdir(val_dir)):
        if img_file.endswith('.JPEG') and img_file in val_annotations:
            img_path = os.path.join(val_dir, img_file)
            class_name = val_annotations[img_file]
            if class_name in class_to_idx:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img).flatten()  # Flatten to 1D
                    x_test_list.append(img_array)
                    y_test_list.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    x_test = np.array(x_test_list, dtype='float32') / 255.0
    y_test = np.array(y_test_list)
    
    # Improved preprocessing: standardize data
    # Compute mean and std for each feature
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8  # Avoid division by zero
    
    # Apply standardization (zero mean, unit variance)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # Convert to JAX arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    
    print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    print(f"Number of classes: {len(class_names)}")
    
    return x_train, y_train, x_test, y_test


def load_mnist(data_dir="../Datasets/mnist"):
    """Load complete MNIST dataset from IDX format files"""
    import struct
    
    def read_idx_images(filename):
        """Read MNIST images from IDX format"""
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
        return images
    
    def read_idx_labels(filename):
        """Read MNIST labels from IDX format"""
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    # Load training data
    x_train = read_idx_images(f"{data_dir}/train-images-idx3-ubyte")
    y_train = read_idx_labels(f"{data_dir}/train-labels-idx1-ubyte")
    
    # Load test data
    x_test = read_idx_images(f"{data_dir}/t10k-images-idx3-ubyte")
    y_test = read_idx_labels(f"{data_dir}/t10k-labels-idx1-ubyte")
    
    # Data preprocessing: normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Improved preprocessing: standardize data
    # Compute mean and std for each feature
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8  # Avoid division by zero
    
    # Apply standardization (zero mean, unit variance)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # Convert to JAX arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    
    print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    
    return x_train, y_train, x_test, y_test


def load_cifar100(data_dir="../Datasets/cifar-100-python"):
    """Load complete CIFAR-100 dataset"""
    # Training data
    train_batch = unpickle(f"{data_dir}/train")
    x_train = train_batch[b'data']
    y_train = np.array(train_batch[b'fine_labels'])  # Use fine labels (100 classes)
    
    # Test data
    test_batch = unpickle(f"{data_dir}/test")
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'fine_labels'])  # Use fine labels (100 classes)
    
    # Data preprocessing: reshape and normalize
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_train = x_train.reshape(-1, 3072).astype('float32') / 255.0
    
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3072).astype('float32') / 255.0
    
    # Improved preprocessing: standardize data
    # Compute mean and std for each feature
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-8  # Avoid division by zero
    
    # Apply standardization (zero mean, unit variance)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # Convert to JAX arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    
    print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    
    return x_train, y_train, x_test, y_test