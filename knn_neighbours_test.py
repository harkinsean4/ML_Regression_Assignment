import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# twelve attributes
attribtues = ['normalising_temperature','tempering_temperature', 'sample', 'percent_silicon', 'percent_chromium', 'manufacture_year', 'percent_copper', 'percent_nickel', 'percent_sulphur', 'percent_carbon', 'percent_manganese', 'tensile_strength']
used_attribtues = ['normalising_temperature','tempering_temperature', 'percent_silicon', 'percent_chromium', 'manufacture_year', 'percent_copper', 'percent_nickel', 'percent_sulphur', 'percent_carbon', 'percent_manganese', 'tensile_strength']

features = []
targets = []

def main():

    file_name = 'steel.txt'

    samples = readInData(file_name)

    # Must predict the tensile strength

    for sample in samples:
        features.append(sample[0:len(attribtues)-2]) 
        targets.append(sample[-1])

    X = np.array(features)  
    y = np.array(targets) 

    X = X.astype(np.float64)
    y = y.astype(np.float64)

    '''Regression'''

    print("There are {} samples".format(len(y)))
    train_size = round((2/3)*len(y))

    # random_state set to zero so we generate the same randomised X and y data every time we run the code
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)

    # When p = 1, this is equivalent to using manhattan_distance (l1)
    # euclidean_distance (l2) for p = 2
    # For arbitrary p, minkowski_distance (l_p) is used.

    max_neigbours = 21

    for i in range(max_neigbours):

        num_neighbours = i + 1
        print("Number of neighbours: {}".format(num_neighbours))

        # When p = 1, this is equivalent to using manhattan_distance (l1)
         # euclidean_distance (l2) for p = 2
        # For arbitrary p, minkowski_distance (l_p) is used.
        knnReg = KNeighborsRegressor(n_neighbors=num_neighbours, p=1)

        knn_reg_model = knnReg.fit(X_train, y_train) 
        knn_reg_accuracy = knn_reg_model.score(X_test, y_test)  # Returns the coefficient of determination R^2 of the prediction.

        knn_y_pred = knn_reg_model.predict(X_test) # Predcitions based off our testing data
        # print("KNN Regression model has an accuracy of {}".format(knn_reg_accuracy))

        print("KNN Regression Mean squared error: %.2f" % mean_squared_error(y_test, knn_y_pred))  # The mean squared error
        print("KNN Regression Variance score: %.2f" % r2_score(y_test, knn_y_pred))               # Explained variance score: 1 is perfect prediction


# func written by Sean Harkin - Also used in Assignment 1 & 2
def readInData(file_name):

    samples = []

    with open(file_name, 'r') as my_file:
        
        print('Reading in Text File')
        my_file.seek(0)
        
        text_file_row_number = 0

        for line in my_file.readlines():
            # read in a full row of the text file, precisely 200 elements per row
            sample = []
            element_num = 0  

            for row_element in line.split('\t'):
            
                row_element = row_element.rstrip('\n') #strip away the new line carrige '\n'

                #omit third element becuase it is the sample id
                if element_num != 2:
                    sample.append(row_element)
                element_num = element_num + 1

            samples.append(sample)
            text_file_row_number = text_file_row_number + 1

        print("File read")    

    return samples


if __name__ == '__main__':
    main()