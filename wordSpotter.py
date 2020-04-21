
import sys
import glob
import argparse
import numpy as np
import math
import cv2
from scipy.stats import multivariate_normal
from sklearn import mixture
import time
from sklearn import svm
from sklearn.decomposition import PCA
import os
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
import timeit
from sklearn.metrics import label_ranking_average_precision_score
import pickle
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import phow
from phoc import PHOC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from cyvlfeat.gmm import gmm as GaussianMixture
from cyvlfeat.fisher import fisher as FisherVector

pca_obj = None
load_gmm_flag = False
svm_obj = None

def calcGaussian(descriptors_i):
    N=16
    gmeans, gcovars, gpriors, ll, pos = GaussianMixture(descriptors_i,n_clusters=N, max_num_iterations=100,n_repetitions=2,verbose=False)
    return (gmeans,gcovars,gpriors)

def dictionary(descriptors, desc_mapping, N):
    '''
    Dictionary of SIFT features using GMM
    '''
    means_ = []
    covariances_ = []
    weights_ = []
    pool = mp.Pool(mp.cpu_count())
    gmms = pool.map(calcGaussian, [np.asarray(descriptors[desc_mapping == i]) for i in range(max(desc_mapping)+1)])
    pool.close()
    pool.join()
    means_ = [gmm[0] for gmm in gmms]
    covariances_ = [gmm[1] for gmm in gmms]
    weights_ = [gmm[2] for gmm in gmms]
    return np.array(means_), np.array(covariances_), np.array(weights_)

def splitImage(im, M=2, N=6):
    # split the image into 12 parts
    im = im.copy()
    new_h = math.ceil(im.shape[0]/M)*M
    new_w = math.ceil(im.shape[1]/N)*N
    im = cv2.resize(im, (max(6,new_w), max(2,new_h)))
    x_offset = math.ceil(im.shape[1]*1.00/N)
    y_offset = math.ceil(im.shape[0]*1.00/M)
    tiles = [im[y:min(y+y_offset,im.shape[0]),x:min(x+x_offset,im.shape[1])] for y in range(0,im.shape[0],y_offset)
                                                    for x in range(0,im.shape[1],x_offset)]
    return tiles

def getImgSegmentDescriptors(img1):
    # Get the interest points in a specific image segment
    # SIFT features are densely extracted using 6 different patch sizes.
    sizes = [2,4,6,8,10,12]
    sizes = [6]
    kp1, des1 = phow.vl_phow(img1, color="gray",sizes=sizes)
    kp1[:,1] = (kp1[:,1]-img1.shape[1]/2)/img1.shape[1]
    kp1[:,0] = (kp1[:,0]-img1.shape[0]/2)/img1.shape[0]
    if(len(kp1) == 0):
        des1 = np.zeros((1,128))
        kp1 = np.zeros((1,2))
    des1 = np.concatenate((des1,kp1), axis=1)
    return des1

def image_descriptors(file):
    # Computing the dense sift matching for a "single image"
    # Divide image into 12 segments. Calculate SIFT descriptors of each segment
    # and create a mapping for each segment. Then concatenate and return them.
    img1 = cv2.imread(file)
    if(img1 is None):
        print("None type image path: {0}".format(file))
        return None
    img_segments = splitImage(img1, 2, 6)
    mapping = []
    descriptors = None
    i = 0
    if (len(img_segments) != 12):
        print("Image segments aren't 12")
    for seg in img_segments:
        temp_descriptors = getImgSegmentDescriptors(seg)
        if(descriptors is None):
            descriptors = temp_descriptors
        else:
            descriptors = np.concatenate((descriptors,temp_descriptors),axis=0)

        mapping += [i]*len(temp_descriptors)
        i = i + 1
    return (np.array(descriptors), np.array(mapping))


def folder_descriptors(folder):
    # Get the SIFT descriptions for all images in a "folder" rescursively
    files = glob.glob(folder + "/*.png")
    print(folder)
    print("Calculating descriptors. Number of images is", len(files))
    res = None
    mapping = None
    for file in files:
        img1 = cv2.imread(file)
        print(file)
        desc, temp_map = image_descriptors(file)
        if desc is not None:
            if res is not None:
                res = np.concatenate((res,desc),axis=0)
                mapping = np.concatenate((mapping,temp_map),axis=0)
            else:
                res = desc
                mapping = temp_map
    return (res,mapping)

def normalize(fisher_vector):
    '''
    Power and L2 Normalization
    '''
    v = np.multiply(np.sqrt(abs(fisher_vector)), np.sign(fisher_vector))
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(words_with_mapping, means, covs, w):
    '''
    Building the FV for a image, sample denotes a list of SIFT feature vectors
    '''
    # global pca_obj
    words = words_with_mapping[0]
    desc_mapping = words_with_mapping[1]
    words = reduceDimensions(words)
    fv = None
    if(len(np.unique(desc_mapping))!=12):
        print("hfjdhfjhdjhfjdhfhjdfjhdjhfjhdjhfdjhf")
    for i in range(max(desc_mapping)+1):
        samples = np.asarray(words[desc_mapping == i])
        samples = np.float32(samples.T)
        #samples = np.reshape(samples,(1,-1))
        means_i = means[i]
        covs_i = covs[i]
        w_i = w[i]
        if(len(samples) == 0):
            print("Zero samples")
        means_i = means_i.T
        covs_i = covs_i.T
        means_i = np.float32(means_i)
        covs_i = np.float32(covs_i)
        w_i = np.float32(w_i)
        fv_i = FisherVector(samples, means_i, covs_i, w_i, normalized=True, fast=True)
        if(fv is None):
            fv = fv_i
        else:
            fv = np.concatenate((fv,fv_i),axis = 0)
    return np.array(fv)


def reduceDimensions(words):
    '''
    Using PCA to reduce dimensions,-,-2,-22
    last two stores coordinate
    '''
    global pca_obj
    global load_gmm_flag
    try:
        if(pca_obj is None):
            pca = PCA(n_components=62)
            pca_obj = pca.fit(words[:,:-2])
            with open("./pca_dump", 'wb') as handle:
                pickle.dump(pca_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res = pca_obj.transform(words[:,:-2])
        res = np.concatenate((res,words[:,-2:]),axis=1)
        return res
    except:
        print("error in Reduce Dimensions")
        print("words shape: {0}".format(words.shape))

# The calculated PCA is stored with the help of pickle, so that it can be loaded without retraining.

def loadPCA(path):
    global pca_obj
    with open("./pca_dump", 'rb') as handle:
        pca_obj = pickle.load(handle)


def generate_gmm(opts, N):
    '''
    Generating the GMM and saving the parameters
    '''
    start = timeit.default_timer()
    pool = mp.Pool(mp.cpu_count())
    words_with_mapping = pool.map(folder_descriptors, [folder for folder in glob.glob(opts.gmm_train_data_path + '*')])
    pool.close()
    pool.join()
    words = np.concatenate([word[0] for word in words_with_mapping])
    word_mapping = np.concatenate([word[1] for word in words_with_mapping])
    stop = timeit.default_timer()
    print('Time taken for getting features: ', stop - start)
    words = reduceDimensions(words)
    print("Training GMM of size", N)
    means, covs, weights = dictionary(words, word_mapping, N)
    #Throw away gaussians with weights that are too small:
    th = 1.0 / N
    th = 0
    for i in range(len(means)):
        means[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), means[i]) if weights[i][k] > th])
        covs[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), covs[i]) if weights[i][k] > th])
        weights[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), weights[i]) if weights[i][k] > th])
    np.save(opts.weights_data_path + "means.gmm", means)
    np.save(opts.weights_data_path + "covs.gmm", covs)
    np.save(opts.weights_data_path + "weights.gmm", weights)
    return means, covs, weights

def get_fisher_vectors_from_folder(gmm, folder):
    '''
    Getting the FVs of all the images in the folder
    '''
    files = glob.glob(folder + "/*.png")
    res = {}
    for file in files:
        temp = image_descriptors(file)
        if(temp is not None):
            # print(temp)
            # print(os.path.basename(file))
            res[os.path.basename(file)] = np.float32(
                fisher_vector(temp, *gmm))
    return res
    # return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])

def fisher_features(folder, gmm):
    '''
    Getting the FVs of all the images in the subfolders in the directory
    '''
    folders = glob.glob(folder + "/*")
    res = {}
    temp_fun = partial(get_fisher_vectors_from_folder, gmm)
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(temp_fun, [f for f in folders])
    pool.close()
    pool.join()
    for result in results:
        res.update(result)
    return res

def get_image_mapping_from_folder(folder):
    '''
    Getting the Image Name to absolute path mapping
    '''
    files = glob.glob(folder + "/*.png")
    res = {}
    for file in files:
        res[os.path.basename(file)] = os.path.abspath(file)
    return res


def get_image_mappings(folder):
    '''
    Getting the Image Name to absolute path mapping recursively
    '''
    folders = glob.glob(folder + "/*")
    res = {}
    for f in folders:
        res.update(get_image_mapping_from_folder(f))
    return res

def calcTrainingPHOC(word_strings_dict):
    word_phoc_dict = {}
    for img, string_repr in word_strings_dict.items():
        word_phoc_dict[img] = PHOC()(string_repr)
    return word_phoc_dict

def calcPHOC(string_repr):
    return PHOC()(string_repr)

def load_gmm(path):
    '''
    Loading GMM
    '''
    print("in load gmm")
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    res = map(lambda file: np.load(file), map(
        lambda s: path + s, files))
    return tuple(res)


def get_word_strings_from_file(file_path):
    '''
    Getting the word strings from the xml filepath
    '''
    res = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    lines = root.findall("./handwritten-part/line")
    for line in lines:
        for word in line.findall('word'):
            id = word.get('id')
            word_string = word.get('text')
            res[id+".png"] = word_string
    return res


def extractWordStrings(folder_path):
    '''
    Extracting the word strings from all the xml files present in the folder
    '''
    word_strings = {}
    folders = glob.glob(folder_path + "*.xml")
    for file in folders:
        word_strings.update(get_word_strings_from_file(file))
    return word_strings

def L2Normalize(v):
    v = np.array(v).copy()
    return np.nan_to_num(v/np.linalg.norm(v, axis=1, keepdims=True))


def MAPScore(query_path, word_strings_dict, phoc_features, gmm, image_mapping_dict, show_img_flag = False, cca_obj=None):
    '''
    Getting the MAP score for the given image query
    '''
    if(show_img_flag):
        img = plt.imread(query_path)
        plt.imshow(img)
        plt.show()
    query_sift_features = image_descriptors(query_path)
    if(query_sift_features is None):
        raise Exception("hello")
    temp = copy.deepcopy(gmm)
    query_FV = fisher_vector(query_sift_features, *temp)
    # print(query_FV)
    query_FV = query_FV.reshape(1, -1)
    phoc = svm_obj.predict(query_FV)
    phoc = phoc*2 - 1
    if(show_img_flag):
        print("path: {0}".format(query_path))
        print(np.unique(phoc, return_counts=True))
    if(cca_obj is not None):
        phoc = cca_obj.transform_a(L2Normalize(phoc))
        phoc = L2Normalize(phoc)
    phoc_values = np.array(list(phoc_features.values()))
    phoc_keys = np.array(list(phoc_features.keys()))
    similarity_score = cosine_similarity(phoc, phoc_values)
    # print(similarity_score.shape)
    max_index = np.argmax(similarity_score)
    top_5_indices = similarity_score.flatten().argsort()[-5:][::-1]
    img = cv2.imread(query_path,0)
    bar = np.zeros((img.shape[0], 5), np.uint8)
    shape = img.shape
    if(show_img_flag):
        print("top 5 indices {0}".format(top_5_indices))
        for i in top_5_indices:
            match_img_path = image_mapping_dict[phoc_keys[i]]
            print("Matching image path: {0}".format(match_img_path))
            print("match phoc values: {0}".format(np.unique(phoc_features[phoc_keys[i]], return_counts=True)))
            img2 = cv2.imread(match_img_path,0)
            plt.imshow(img2)
            plt.show()
            img = np.hstack((img, bar, cv2.resize(img2, (shape[1],shape[0]))))
        cv2.imwrite(str(query_path).split("/")[-1] + '_output.png', img)
    query_string = word_strings_dict[os.path.basename(query_path)]
    word_vals = np.array([word_strings_dict[your_key]
                          for your_key in phoc_features.keys()])
    word_vals = word_vals.flatten()
    y_true = np.array([[int(1) if s == query_string else int(0)
                        for s in word_vals]])
    similarity_score[similarity_score <=0] = 0
    mape = label_ranking_average_precision_score(y_true, similarity_score)
    print(mape)
    return mape


def QBS(query_string, word_strings_dict, phoc_features, gmm, image_mapping_dict, show_img_flag = False, cca_obj=None):
    '''
    Getting the MAP score for the given image query
    '''

    phoc = np.array(PHOC()(query_string)).reshape(1, -1)
    if(cca_obj is not None):
        phoc = cca_obj.transform_b(L2Normalize(phoc))
        phoc = L2Normalize(phoc)
    if(show_img_flag):
        print(np.unique(phoc, return_counts=True))
    phoc_values = np.array(list(phoc_features.values()))
    phoc_keys = np.array(list(phoc_features.keys()))
    similarity_score = cosine_similarity(phoc, phoc_values)
    max_index = np.argmax(similarity_score)
    top_5_indices = similarity_score.flatten().argsort()[-5:][::-1]
    shape = None
    if(show_img_flag):
        print("top 5 indices {0}".format(top_5_indices))
        for i in top_5_indices:
            match_img_path = image_mapping_dict[phoc_keys[i]]
            print("Matching image path: {0}".format(match_img_path))
            print("match phoc values: {0}".format(np.unique(phoc_features[phoc_keys[i]], return_counts=True)))
            img2 = cv2.imread(match_img_path,0)
            plt.imshow(img2)
            plt.show()
            if(shape is None):
                img = img2
                bar = np.zeros((img.shape[0], 5), np.uint8)
                shape = img.shape
            else:
                print(img.shape, bar.shape, cv2.resize(img2, img.T.shape).shape)
                img = np.hstack((img, bar, cv2.resize(img2, (shape[1],shape[0]))))
        cv2.imwrite(str(query_path).split("/")[-1] + '_output.png', img)
    word_vals = np.array([word_strings_dict[your_key]
                          for your_key in phoc_features.keys()])
    word_vals = word_vals.flatten()
    y_true = np.array([[int(1) if s == query_string else int(0)
                        for s in word_vals]])
    similarity_score[similarity_score <=0] = 0
    mape = label_ranking_average_precision_score(y_true, similarity_score)
    return mape

def score(word_strings_dict, train_phoc, gmm, image_mapping_dict, folder):
    try:
        print("I'm running on CPU {0}".format(mp.current_process().name))
    except:
        print("I'm running on CPU {0}".format(os.getpid()))
    image_paths = glob.glob(folder + "/*.png")
    score_list = []
    for img_path in image_paths:
        # print("count: {0}".format(count))
        score = MAPScore(img_path, word_strings_dict,
                        train_phoc, gmm, image_mapping_dict, False)
        score_list.append(score)
    return np.array(score_list)


class Params():
    def __init__(self, gmm_train_data_path, svm_train_data_path, xml_data_path, weights_data_path, model_data_dump_path):
        self.gmm_train_data_path = gmm_train_data_path
        self.svm_train_data_path = svm_train_data_path
        self.xml_data_path = xml_data_path
        self.weights_data_path = weights_data_path
        self.model_data_dump_path = model_data_dump_path

if __name__ == "__main__":
    gmm_train_data_path = "dataset/gmmTrain/"
    svm_train_data_path = "dataset/SVMTrain/"
    xml_data_path = "dataset/xml/"
    weights_data_path = "dataset/weights/"
    model_data_dump_path = "dataset/modelsdump/"


    opts = Params(gmm_train_data_path, svm_train_data_path,xml_data_path, weights_data_path, model_data_dump_path)

    no_gaussians = 16
    print("no. of weights {0}".format(no_gaussians))
    start = timeit.default_timer()
    # Load/generate GMM based on load_gmm_flag
    # gmm is a tuple of size 3 containing means, covs and weights of the 16 GMMs
    gmm = load_gmm(opts.weights_data_path) if load_gmm_flag else generate_gmm(
        opts, no_gaussians)
    print(gmm)
    print(len(gmm))
    stop = timeit.default_timer()
    print('Time taken for training GMM: ', stop - start)


    if(load_gmm_flag):
        loadPCA(opts.weights_data_path)

    svm_FV_features = None
    gmm_FV_features = None

    print("Getting Fisher Vector encoding of training data")
    start = timeit.default_timer()
    if(load_gmm_flag):
        with open(opts.weights_data_path + "svm_train_FV_dump", 'rb') as handle:
            svm_FV_features = pickle.load(handle)
        with open(opts.weights_data_path + "gmm_train_FV_dump", 'rb') as handle:
            gmm_FV_features = pickle.load(handle)
    else:
        svm_FV_features = fisher_features(opts.svm_train_data_path, gmm)
        print(len(svm_FV_features))
        with open(opts.weights_data_path + "svm_train_FV_dump", 'wb') as handle:
            pickle.dump(svm_FV_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gmm_FV_features = fisher_features(opts.gmm_train_data_path, gmm)
        with open(opts.weights_data_path + "gmm_train_FV_dump", 'wb') as handle:
            pickle.dump(gmm_FV_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Build a dictionary full_data_FV_features using update() method.   
    full_data_FV_features = copy.deepcopy(svm_FV_features)
    full_data_FV_features.update(svm_FV_features)

    stop = timeit.default_timer()
    print('Time taken for getting FV encodings: ', stop - start)

    print("Getting word strings from xml data")
    ### img - str repr
    start = timeit.default_timer()
    word_strings_dict = None
    if(load_gmm_flag):
        with open(opts.xml_data_path + "word_string_dict_dump", 'rb') as handle:
            word_strings_dict = pickle.load(handle)
    else:
        word_strings_dict = extractWordStrings(opts.xml_data_path)
        with open(opts.xml_data_path + "word_string_dict_dump", 'wb') as handle:
            pickle.dump(word_strings_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    stop = timeit.default_timer()
    print('Time taken for getting xml encodings: ', stop - start)

    ## Getting image mapping dict
    ### img - path
    print("Getting word strings mappings")
    start = timeit.default_timer()
    image_mapping_dict = get_image_mappings(opts.gmm_train_data_path)
    image_mapping_dict.update(get_image_mappings(opts.svm_train_data_path))
    stop = timeit.default_timer()
    print('Time taken for get word string mappings: ', stop - start)

    print("Getting train PHOC")
    ## str phoc of all words in xml
    start = timeit.default_timer()
    train_phoc = calcTrainingPHOC(word_strings_dict)
    stop = timeit.default_timer()
    print('Time taken to get train PHOC encoding: ', stop - start)
    print("Build smv/cca train dataset")
    start = timeit.default_timer()
    X, Y, _ = buildDataset(svm_FV_features, train_phoc)
    cca_X, cca_Y, _ = buildDataset(cca_FV_features, train_phoc)
    full_X, full_Y, img_names = buildDataset(full_data_FV_features, train_phoc)
    print("svm train phoc encoding")
    print(np.unique(Y, return_counts=True))
    stop = timeit.default_timer()
    print('Time taken to build train dataset: ', stop - start)

    print("Training SVM")
    start = timeit.default_timer()
    if(load_gmm_flag):
        with open(opts.model_data_dump_path + "svm_obj", 'rb') as handle:
            svm_obj = pickle.load(handle)

    if(svm_obj is None):
        clf = SGDClassifier(alpha=0.0001, eta0=0.003, tol=1e-5,class_weight="balanced", n_jobs=-1)
        svm_obj = OneVsRestClassifier(clf, n_jobs=-1)
        svm_obj.fit(X, Y)
        with open(opts.model_data_dump_path + "svm_obj", 'wb') as handle:
            pickle.dump(svm_obj, handle,protocol=pickle.HIGHEST_PROTOCOL)

    stop = timeit.default_timer()
    print('Time taken to train SVM: ', stop - start)

    cca_Y_pred = None
    if(load_gmm_flag):
        cca_Y_pred = np.load(opts.weights_data_path + "cca_Y_pred" + ".npy")

    if(cca_Y_pred is None):
        cca_Y_pred = svm_obj.predict(cca_X)
        print("svm prediction for cca_X") 
        print(np.unique(cca_Y_pred,return_counts=True))
        cca_Y_pred = cca_Y_pred*2 - 1
        np.save(opts.weights_data_path + "cca_Y_pred", cca_Y_pred)

    full_Y_pred = None
    if(load_gmm_flag):
        full_Y_pred = np.load(opts.weights_data_path + "full_Y_pred" + ".npy")

    if(full_Y_pred is None):
        full_Y_pred = svm_obj.predict(full_X)
        print("svm prediction for full data") 
        print(np.unique(full_Y_pred,return_counts=True))
        full_Y_pred = full_Y_pred*2 - 1
        np.save(opts.weights_data_path + "full_Y_pred", full_Y_pred)

    v = full_Y_pred
    cca_obj = None
    full_data_phoc = dict((key, value) for (key, value) in zip(img_names, v))

    scores = []
    while(True):
        query_type = input(
            "Press 1 for get test MAPScore with cca\nPress 2 for get test baselineMAPScore\nPress 3 for string\nPress 4 for single image\nPress 0 to exit\n")
        if(int(query_type) == 0):
            break
        if(int(query_type) == 1):
            start = timeit.default_timer()
            score_list = []
            test_data_path = input("Enter query images folder path: ")
            folders = glob.glob(test_data_path + "*")
            count = 0
            for folder in folders:
                image_paths = glob.glob(folder + "/*.png")
                for img_path in image_paths:
                    count += 1
                    if(count%400 == 0):
                        print("count: {0}".format(count))
                        print("temp MAP Score: {0}".format(np.mean(score_list)))
                    try:
                    # print("count: {0}".format(count))
                        score = MAPScore(img_path, word_strings_dict,
                                        full_data_phoc, gmm, image_mapping_dict, False, cca_obj)
                        if(math.isfinite(score)):
                            score_list.append(score)
                    except:
                        print("seg fault path: {0}".format(img_path))
            try:
                score_list = np.array(score_list)
                print("MAP Score: {0}".format(np.mean(score_list)))
            except:
                print(len(score_list))
            stop = timeit.default_timer()
            print('Time taken to get test MAP: ', stop - start)
        elif(int(query_type) == 2):
            start = timeit.default_timer()
            score_list = []
            test_data_path = input("Enter query images folder path: ")
            folders = glob.glob(test_data_path + "*")
            count = 0
            for folder in folders:
                image_paths = glob.glob(folder + "/*.png")
                for img_path in image_paths:
                    count += 1
                    if(count%200 == 0):
                        print("count: {0}".format(count))
                        print("temp MAP Score: {0}".format(np.mean(score_list)))
                    score = MAPbaseLineScore(img_path, word_strings_dict,
                                    full_data_FV_features, gmm, image_mapping_dict, False)
                    if(math.isfinite(score)):
                        score_list.append(score)
            try:
                score_list = np.array(score_list)
                print("MAP Score: {0}".format(np.mean(score_list)))
            except:
                print(len(score_list))
            stop = timeit.default_timer()
            print('Time taken to get test MAP: ', stop - start)
        elif(int(query_type) == 3):
            query_path = input("Enter query string: ")
            score = QBS(query_path, word_strings_dict,
                            full_data_phoc, gmm, image_mapping_dict, True, cca_obj)
            scores.append(score)
            print("MAP Score: {0}".format(score))
        elif(int(query_type) == 4):
            query_path = input("Enter query image path: ")
            if(query_path == "break"):
                break
            try:
                score = MAPScore(query_path, word_strings_dict,
                                full_data_phoc, gmm, image_mapping_dict, True)
                scores.append(score)
            except:
                print(query_path)
            print("MAP Score: {0}".format(score))
        else:
            print("MAP Score: {0}".format(np.mean(score_list)))
