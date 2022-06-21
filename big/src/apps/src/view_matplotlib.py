import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


from src.view_abc import View


class ViewMatplotlib(View):
    def __init__(self) -> None:
        super().__init__()
    

    def tabular_images(self, np_imnages, np_label, size=(10,10)) -> None:
        np_arr_shuffle, np_label_shuffle = shuffle(np_imnages, np_label, random_state=42)
        fig = plt.figure(figsize=(30, 30))
        rows = size[0]
        columns = size[1]  
        ax = []
        for i in range(1, columns*rows +1):
            img = np_arr_shuffle[i]
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title(str(np_label_shuffle[i]))  # set title
            ax[-1].grid(False)
            ax[-1].axis('off')
            plt.imshow(img)
        plt.show()

    
    def __1d_to_2d(self,n):
        if n==1:
            cols = rows = 1
            fig = plt.figure(figsize=(8, 8))  
        elif n==2:
            cols = 2
            rows = 1
            fig = plt.figure(figsize=(8, 8))    
        elif n==3 or n==4:
            cols = 2
            rows = 2
            fig = plt.figure(figsize=(8, 8))    
        elif n==5 or n==6:
            cols = 3
            rows = 2
            fig = plt.figure(figsize=(8, 6))    
        elif n==7 or n==8:
            cols = 4
            rows = 2 
            fig = plt.figure(figsize=(16, 8))     
        elif n==9:
            cols = 3
            rows = 3    
            fig = plt.figure(figsize=(16, 16))  
        elif n==10:
            cols = 5
            rows = 2    
            fig = plt.figure(figsize=(16, 8))         
        elif n==11 or n==12:
            cols = 4
            rows = 3    
            fig = plt.figure(figsize=(16, 12))     
        elif n==13 or n==14 or n==15:
            cols = 5
            rows = 3    
            fig = plt.figure(figsize=(16, 10))   
        elif n==17 or n==18 or n==19 or n==20:
            cols = 5
            rows = 4    
            fig = plt.figure(figsize=(16, 14))
        elif n==21 or n==22 or n==23 or n==24 or n==25:
            cols = 5
            rows = 5
            fig = plt.figure(figsize=(16, 18))      
        elif n==26 or n==27 or n==28 or n==29 or n==30:
            cols = 5
            rows = 6    
            fig = plt.figure(figsize=(16, 20))      
        else:
            cols = 5
            rows = 7    
            fig = plt.figure(figsize=(12, 20))      
        return (cols, rows, fig)


    def double_title_tabular_images(self, np_images, np_label, np_predict_label) -> None:
        size_image = np_images[0].shape
        cols, rows, fig = self.__1d_to_2d(len(np_images))

        fig = plt.figure(figsize=(30, 30))
        ax = []
        for i in range(1, cols*rows+1):
            ax.append(fig.add_subplot(rows, cols, i))
            if len(np_images)>i-1:
                ax[-1].set_title("Pred:{}\nTrue:{}".format(np_predict_label[i-1], np_label[i-1]))
                ax[-1].grid(False)
                ax[-1].axis('off')
                img = np_images[i-1]
                plt.imshow(img)
            else:
                ax[-1].set_title("Pred:{}\nTrue:{}".format(None, None))
                ax[-1].grid(False)
                ax[-1].axis('off')
                img = np.zeros((size_image[0], size_image[1], size_image[2]), dtype=int)
                plt.imshow(img)
        plt.show()


    def tabular_skelets(self, np_skelets,  np_label, size=(10,10)) -> None:
        np_arr_shuffle, np_label_shuffle = shuffle(np_skelets, np_label, random_state=42)
        ax = []
        # cols, rows, fig = self.__1d_to_2d(len(np_arr_shuffle[:6]))
        fig = plt.figure(figsize=(30, 30))
        (cols, rows) = size
        for i in range(1, cols*rows +1):
            ax.append(fig.add_subplot(rows, cols, i))
            self.plot_skelet(ax[-1], np_arr_shuffle[i-1], np_label_shuffle[i-1])
        plt.show()

    
    def double_title_tabular_skelets(self, np_images, np_label, np_predict_label, size=(10,10)) -> None:
        ax = []
        cols, rows, fig = self.__1d_to_2d(len(np_images))
        #fig = plt.figure(figsize=(30, 30))
        #(cols, rows) = size
        for i in range(1, cols*rows +1):
            ax.append(fig.add_subplot(rows, cols, i))
            self.double_title_plot_skelet(ax[-1], np_images[i-1], np_label[i-1], np_predict_label[i-1])
        plt.show()

    def double_title_plot_skelet(self, ax, np_skelet, label, predict_label) -> None:
        x = np_skelet[:,0]
        y = np_skelet[:,1]*(-1)
        #fig, ax = plt.subplots()
        ax.set_title("Pred:{}\nTrue:{}".format(predict_label, label))
        ax.grid(False)
        ax.axis('off')
        for n in self.list_line_point_hand:
            ax.plot([x[n[0]], x[n[1]]], [y[n[0]], y[n[1]]], c="red")
        ax.scatter(x, y, c="blue")

    def plot_skelet(self, ax, np_skelet, label) -> None:
        x = np_skelet[:,0]
        y = np_skelet[:,1]*(-1)
        #fig, ax = plt.subplots()
        ax.set_title("Label:{}".format(label))
        ax.grid(False)
        ax.axis('off')
        for n in self.list_line_point_hand:
            ax.plot([x[n[0]], x[n[1]]], [y[n[0]], y[n[1]]], c="red")
        ax.scatter(x, y, c="blue")

    def hand_model3d(self, np_arr1) -> None:

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        hand_line = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
                    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],
                    [14,15],[0,17],[17,18],[18,19],[19,20]]
        for n in hand_line:
            zline = [np_arr1[n[0],0], np_arr1[n[1],0]]
            xline = [np_arr1[n[0],1], np_arr1[n[1],1]]
            yline = [np_arr1[n[0],2], np_arr1[n[1],2]]
            ax.plot3D(xline, yline, zline, 'gray')

        # Data for three-dimensional scattered points
        zdata = np_arr1[:,0]
        xdata = np_arr1[:,1]
        ydata = np_arr1[:,2]
        ax.scatter3D(xdata, ydata, zdata);    

    def double_bar_comparison(self, not_corr_i, X_test, Y_test, Y_pred):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.double_bar_comparison_plot(fig, ax, not_corr_i, X_test, Y_test, Y_pred)
        plt.show()

    def double_bar_comparison_tabular(self, X_test, Y_test, Y_pred):
        not_correct = np.nonzero(Y_test != Y_pred)[0]
        ax = []
        cols, rows, fig = self.__1d_to_2d(len(not_correct))
        #fig = plt.figure(figsize=(30, 30))
        #(cols, rows) = size

        for i in range(1, cols*rows +1):
            ax.append(fig.add_subplot(rows, cols, i))
            #self.double_bar_comparison_plot(fig, ax[-1], X_test[i-1], Y_test[i-1], np_predict_label[i-1])

            self.double_bar_comparison_plot(fig, ax[-1], i-1, X_test, Y_test, Y_pred)
        plt.show()
    
    def double_bar_comparison_plot(self, fig, ax, not_corr_i, X_test, Y_test, Y_pred):
        not_correct = np.nonzero(Y_test != Y_pred)[0]
        
        list_i_pred_correct = []
        X_test_by_not_correct = []
        for i, n in enumerate(Y_test):
            if n==Y_test[not_correct[not_corr_i]] and not(i in not_correct):
                X_test_by_not_correct.append(X_test[i])
                # print("Y_test[i]",Y_test[i]) 
                list_i_pred_correct.append(i)
        a = np.array(X_test_by_not_correct)
        current = X_test[not_correct[not_corr_i]]
        #print("a", a)
        max_ = np.max(a, axis=0)
        mean_ = np.mean(a, axis=0)
        min_  = np.min(a, axis=0)
        feature = [str(i) for i in range(X_test.shape[1])]
        x = np.arange(len(feature))  # the label locations
        width = 0.4  # the width of the bars
        ax.bar(x + width/2, max_, width, label='Max', color='0.8')
        ax.bar(x + width/2, mean_, width, label='Mean', color='0.6')
        ax.bar(x + width/2, min_, width, label='Min', color='0.4')
        bar_colors = []
        for i, n in enumerate(current):
            if n > min_[i] and n < max_[i]:
                bar_colors.append("green")
            else:
                bar_colors.append("red")
        ax.bar(x - width/2, current, width, label='current', color=bar_colors)
        ax.set_ylabel('Angle')
        ax.set_title("True:{}\nPred:{}".format(Y_test[not_correct[not_corr_i]], Y_pred[not_correct[not_corr_i]]))
        ax.set_xticks(x, feature)
        ax.legend()
        fig.tight_layout()