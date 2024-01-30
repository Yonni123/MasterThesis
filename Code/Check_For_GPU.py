import tensorflow as tf

# Check if TensorFlow is using GPU
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('GPU is available')
    for gpu_device in gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_device)
        print("GPU Device Name:", details['device_name'])
else:
    print("TensorFlow is using CPU.")
