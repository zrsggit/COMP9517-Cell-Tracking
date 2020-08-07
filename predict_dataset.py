import os
import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

from models import create_model

BATCH_SIZE = 8
MARKER_THRESHOLD = 240


def hist_equalization(image):
    return cv2.equalizeHist(image) / 255


def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization
    else:
        return None



def get_image_size(path):
    """returns size of the given image"""

    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


# read images
def load_images(path, cut=False, new_mi=0, new_ni=0, normalization='HE'):

    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    total = len(names)
    normalization_fce = get_normal_fce(normalization)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    for i, name in enumerate(names):

        o = read_image(os.path.join(path, name))

        if o is None:
            print('image {} was not loaded'.format(name))

        image_ = normalization_fce(o)

        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_

    if cut:
        image = image[:, dm:mi16+dm, dn:ni16+dn, :]
    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image

# postprocess markers
def postprocess_markers2(img, threshold=240, erosion_size=12, step=4):

    # distance transform | only for circular objects
    
    # threshold
    m = img.astype(np.uint8)
    _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

    # filling gaps
    hol = binary_fill_holes(new_m*255).astype(np.uint8)

    # morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    new_m = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(new_m)

    return idx, res


def threshold_and_store(predictions,
                        input_images,
                        res_path,
                        thr_markers=240,
                        thr_cell_mask=230,
                        erosion_size=12,
                        step=4,
                        border=0):

    print(predictions.shape)
    print(input_images.shape)
    for i in range(predictions.shape[0]):

        m = predictions[i, :, :, 1] * 255
        # postprocess the result of prediction
        idx, markers = postprocess_markers2(m,
                                            threshold=thr_markers,
                                            erosion_size=erosion_size,
                                            step=step)
        cv2.imwrite('{}/markers{:03d}.tif'.format(res_path, i), markers.astype(np.uint8) * 16)

def predict_dataset(sequence):
    """
    reads images from the path and converts them to the np array
    """
    name = 'DIC-C2DH-HeLa'
    dataset_path = name

    # check if there is a model for this dataset
    print(dataset_path)

    erosion_size = 8
    NORMALIZATION = 'HE'
    # [marker_threshold, cell_mask_threshold]
    MARKER_THRESHOLD, C_MASK_THRESHOLD = [240, 216]
    STEP = 0
    BORDER = 15

    model_name = ['unet_model240_nord_s12_0911_PREDICT.h5']
    print(model_name)
    model_init_path = os.path.join(dataset_path, model_name[0])


    store_path = os.path.join('..', name, '{}_RES'.format(sequence))
    
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    img_path = os.path.join('..', name, sequence)
    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()


    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    print(mi, ni)
    print(new_mi, new_ni)

    input_img = load_images(img_path,
                            new_mi=new_mi,
                            new_ni=new_ni,
                            normalization=NORMALIZATION,
                            )


    model = create_model(model_init_path, new_mi, new_ni)

    pred_img = model.predict(input_img, batch_size=BATCH_SIZE)
    print('pred shape: {}'.format(pred_img.shape))

    org_img = load_images(img_path)
    pred_img = pred_img[:, :mi, :ni, :]

    threshold_and_store(pred_img,
            org_img,
            store_path,
            thr_markers=MARKER_THRESHOLD,
            thr_cell_mask=C_MASK_THRESHOLD,
            erosion_size=erosion_size,
            step=STEP,
            border=BORDER)


if __name__ == "__main__":
    for seq in ["Sequence 1","Sequence 2","Sequence 3","Sequence 4"]:
        predict_dataset(sequence=seq)
