import networkx as nx
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from keras import Input
from keras import backend as K
from keras.models import Model
from matplotlib import pyplot as plt
from skimage import segmentation, color, filters
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import sobel
from skimage import graph
from skimage.io import imread
from spektral.layers import MinCutPool
from tqdm import tqdm
import scipy.sparse as sp
import streamlit as st
import matplotlib.pyplot as plt
#from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray
import cv2
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
image=st.file_uploader('Upload an image',type=['jpg','png','jprg'])
if st.button('Upload'):
    if image is not None:
        st.success('Successfully image is uploaded')
        st.write('Wait for the results....................')
        img=Image.open(image)
        final_img=np.array(img)
        width =  512
        height =  512
        dim = (width,height)

        img_re = cv2.resize(final_img, dim, interpolation = cv2.INTER_AREA)
    

        img_gray= rgb2gray(img_re)
        def weight_boundary(graph, src, dst, n):
    
            default = {'weight': 0.0, 'count': 0}

            count_src = graph[src].get(n, default)['count']
            count_dst = graph[dst].get(n, default)['count']

            weight_src = graph[src].get(n, default)['weight']
            weight_dst = graph[dst].get(n, default)['weight']

            count = count_src + count_dst
            return {
                'count': count,
                'weight': (count_src * weight_src + count_dst * weight_dst) / count
            }


        def merge_boundary(graph, src, dst):
        
            pass


        OVER_SEG = "felzen"
        PLOTS_ON = True
        ALGO = "GNN"
        ITER = 1000
        n_clust = 4
        ACTIV = None
        H_ = None

        # DATA
        #img = imread('/content/output1.jpg')
        #img1 = imread('/content/M42_01_00.jpg')

        # BUILD GRAPH
        # Hyper-segmentation
        if OVER_SEG == "slic":
            segments = segmentation.slic(img_gray, compactness=3, n_segments=200, sigma=1)
        elif OVER_SEG == "felzen":
            segments = segmentation.felzenszwalb(img_gray, scale=300, sigma=1.0, min_size=50)
        elif OVER_SEG == "quick":
            segments = segmentation.quickshift(gray2rgb(img_gray), kernel_size=3, max_dist=6, ratio=0.5)
        elif OVER_SEG == "water":
            gradient = sobel(rgb2gray(img_gray))
            segments = segmentation.watershed(gradient, markers=400, compactness=0.001)
            segments -= 1
        else:
            raise ValueError(OVER_SEG)

        # Region Adjacency Graph
        if ALGO == "hier":
            edges = filters.sobel(color.rgb2gray(img_gray))
            g = graph.rag_boundary(segments, edges)
            labels = graph.merge_hierarchical(segments, g, thresh=0.08, rag_copy=True,
                                            in_place_merge=True,
                                            merge_func=merge_boundary,
                                            weight_func=weight_boundary)
        elif ALGO == "ncut":
            g = graph.rag_mean_color(img_gray, segments, mode='similarity')
            labels = graph.cut_normalized(segments, g, thresh=0.0002, num_cuts=20, in_place=False)
        elif ALGO == "thresh":
            g = graph.rag_mean_color(img_gray, segments, mode='distance')
            labels = graph.cut_threshold(segments, g, 30, in_place=False)
        elif ALGO == "GNN":
            g = graph.rag_mean_color(img_gray, segments, mode='similarity')
            adj = nx.adjacency_matrix(g)
            A = sp.csr_matrix(adj)
            #A = nx.to_scipy_sparse_matrix(g)
            X_m = np.empty((A.shape[0], 3))
            #print(A.shape[0])
            X_t = np.empty((A.shape[0], 3))
            y = np.empty((A.shape[0],))
            for n, d in g.nodes(data=True):
                #print(n, d['mean color'])
                X_m[n] = d['mean color']
                X_t[n] = d['total color']
                y[n] = d['labels'][0]
            X_m = (X_m / np.max(X_m)).astype(np.float32)
            X_t = (X_t / np.max(X_t)).astype(np.float32)
            X = np.concatenate((X_m, X_t), axis=-1)

            n_feat = X.shape[1]
            X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_in'))
            A_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, None)), name='A_in')
            S_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))

            pool1,  seg1, C = MinCutPool(n_clust, activation=ACTIV, h=H_)([X_in, A_in, S_in])

            model = Model([X_in, A_in, S_in], [pool1, seg1])
            model.compile('adam', None ,  metrics=['accuracy'])

            # Setup
            sess = K.get_session()
            loss = model.total_loss
            opt = tf.train.AdamOptimizer(learning_rate=1e-3)
            train_step = opt.minimize(loss)

            # Initialize all variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Fit layer
            tr_feed_dict = {X_in: X,
                            A_in: A.todense(),  # sp_matrix_to_sp_tensor_value(A),
                            S_in: y}
            layer_out = [sess.run([loss], feed_dict=tr_feed_dict)[0]]
            try:
                for _ in tqdm(range(ITER)):
                    outs = sess.run([train_step, loss], feed_dict=tr_feed_dict)
                    layer_out.append(outs[1])
                    x_pool_, seg_pool_ = sess.run([model.output], feed_dict=tr_feed_dict)[0]
            except KeyboardInterrupt:
                st.write('training interrupted!')
            if PLOTS_ON:
                fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,5))
                ax.plot(layer_out, label='Unsupervised Loss')
                ax.legend()
                ax.set_ylabel('Loss')
                ax.set_xlabel('Iteration')
                st.pyplot(fig)


            C_ = sess.run([C], feed_dict=tr_feed_dict)[0]
            c = np.argmax(C_, axis=-1)
            #print("C" , c)
            #print(segments)

            labels = C_[segments]
        else:
            raise ValueError(ALGO)
        adj = nx.adjacency_matrix(g)
        A = sp.csr_matrix(adj)
        #print(len(g.nodes), 'nodes')

        if PLOTS_ON:
            col1,col2,col3=st.columns(3)
            out_seg = color.label2rgb(segments, img_gray, kind='avg')
            out_seg_bound = segmentation.mark_boundaries(out_seg, segments, (0, 0, 0))
            out_clust = color.label2rgb(labels, img_gray, kind='avg')
            fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))
            ax[0].imshow(out_seg)
            ax[0].set_title('Oversegmentation', fontsize=15)
            ax1 = graph.show_rag(segments, g, img_re, border_color=None, img_cmap='gray', edge_cmap='magma', ax=ax[1])
            plt.colorbar(ax1, ax=ax[1])
            ax[1].set_title('Region Adjacency Graph', fontsize=15)
            ax[2].imshow(out_clust)
            ax[2].set_title('MinCutPool', fontsize=15)
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            
            st.pyplot(fig)
    else:
        st.warning('Please upload image')

   


