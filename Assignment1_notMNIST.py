# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import hashlib
import time
from PIL import Image
import imagehash
from sklearn import metrics

# Config the matplotlib backend as plotting inline in IPython
# matplotlib inline

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names



url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere
num_classes = 10
np.random.seed(133)
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# Problem 1
# Let's take a peek at some of the data to make sure it looks sensible. 
# Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.

def problem1(folder_path):
    for file in os.listdir(folder_path):
        d = os.path.join(folder_path, file)
        if os.path.isdir(d):
            img = mpimg.imread(os.path.join(d,os.listdir(d)[random.randint(0, len(d))]))
            imgplot = plt.imshow(img)
            plt.show()

problem1('notMNIST_large')
problem1('notMNIST_small')

# Problem 2
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray.
# Hint: you can use matplotlib.pyplot.

def problem2(folder_path):
  for file in os.listdir(folder_path):
    if file.endswith(".pickle"):
      figx = pickle.load(open(os.path.join(folder_path,file),'rb'))
      plt.imshow(figx[0,:,:])
      plt.title(file)
      plt.show()
      
problem2('notMNIST_large')
problem2('notMNIST_small')

# Problem 3
# Another check: we expect the data to be balanced across classes. Verify that.

def problem3(folder_path):
  for file in os.listdir(folder_path):
    if file.endswith(".pickle"):
      figx = pickle.load(open(os.path.join(folder_path,file),'rb'))
      print(os.path.join(folder_path,file), len(figx), ' images')

problem3('notMNIST_large')
problem3('notMNIST_small')

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# Problem 4
# Convince yourself that the data is still good after shuffling!
def problem4(data,labels):
  label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

  # plot some images to make sure that the labels correspond to images 
  items = random.sample(range(len(labels)), 8)
  for i, item in enumerate(items):
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.title(label_map[labels[item]])
    plt.imshow(data[item])
  plt.show()

problem4(train_dataset, train_labels)
problem4(test_dataset, test_labels)
problem4(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# Problem5
# By construction, this dataset might contain a lot of overlapping samples, 
# including training data that's also contained in the validation and test set! 
# Overlap between training and test can skew the results if you expect to use your model in 
# an environment where there is never an overlap, but are actually ok if you expect to see 
# training samples recur when you use it. Measure how much overlap there is between training, 
# validation and test samples.
# Optional questions:
# What about near duplicates between datasets? (images that are almost identical)
# Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.

def problem5(dataset1, dataset2):
  # dataset1_hash = [hashlib.sha256(img).hexdigest() for img in dataset1]
  # dataset2_hash = [hashlib.sha256(img).hexdigest() for img in dataset2]
  # overlap = 0
  # for i, hash1 in enumerate(dataset1_hash):
  #   for j, hash2 in enumerate(dataset2_hash):
  #     if hash1 == hash2:
  #       overlap = overlap+1
  overlap = {}
  for i, img_1 in enumerate(dataset1):
    for j, img_2 in enumerate(dataset2):     
      if np.array_equal(img_1, img_2):
        overlap[i] = [j]
        break
  return overlap

# print('Overlap between train and test dataset is: ', problem5(train_dataset,test_dataset))
# print('Overlap between train and validation dataset is: ', problem5(train_dataset,valid_dataset))


def image_array_to_diff_hash(ndarr):
    with_pixel_value = ndarr * pixel_depth + pixel_depth/2
    restored_image = Image.fromarray(np.uint8(with_pixel_value))
    return str(imagehash.dhash(restored_image))

def hash_dataset(images):
    return [image_array_to_diff_hash(x) for x in images]

    
all_valid_hashes = hash_dataset(valid_dataset)
all_train_hashes = hash_dataset(train_dataset)
all_test_hashes = hash_dataset(test_dataset)

valid_in_train = np.in1d(all_valid_hashes, all_train_hashes)
test_in_train  = np.in1d(all_test_hashes,  all_train_hashes)
test_in_valid  = np.in1d(all_test_hashes,  all_valid_hashes)

valid_keep = ~valid_in_train
test_keep  = ~(test_in_train | test_in_valid)

valid_dataset_clean = valid_dataset[valid_keep]
valid_labels_clean = valid_labels[valid_keep]

test_dataset_clean = test_dataset[test_keep]
test_labels_clean = test_labels[test_keep]

print("valid -> train overlap: %d samples" % valid_in_train.sum())
print("test  -> train overlap: %d samples" % test_in_train.sum())
print("test  -> valid overlap: %d samples" % test_in_valid.sum())

# Problem 6
# Let's get an idea of what an off-the-shelf classifier can give you on this data.
# It's always good to check that there is something to learn, and that it's a problem that is 
# not so trivial that a canned solution solves it.
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. 
# Hint: you can use the LogisticRegression model from sklearn.linear_model.
# Optional question: train an off-the-shelf model on all the data!

train_samples, _, _ = train_dataset.shape
test_samples, _, _ = test_dataset.shape

def problem6(samples):
  start = time.clock()
  lg = LogisticRegression(n_jobs=-1,verbose=1,multi_class='multinomial', solver='lbfgs',max_iter=1000)
  # Prepare testing data
  X_test = np.reshape(test_dataset,(test_samples,image_size*image_size))
  y_test = test_labels
  # Prepare training data
  X_train = np.reshape(train_dataset[0:samples,:,:],(samples,image_size*image_size))
  y_train = train_labels[0:samples]
  # Train
  lg.fit(X_train, y_train)
  # Predict
  y_pred = lg.predict(X_test)
  accuracy = metrics.accuracy_score(y_test, y_pred)
  # Print results
  print('{} samples, execution time: {}, accuracy: {}.'.format(samples, time.clock()-start, accuracy))

problem6(50)