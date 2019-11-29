import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import time

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

    print('\nLinear Regression\n')

    lreg = LinearRegression()
    lreg_model = lreg.fit(X_train, y_train) 
    lreg_accuracy = lreg_model.score(X_test, y_test)

    y_pred = lreg_model.predict(X_test) # Predcitions based off our testing data
    print("Linear Regression model has an accuracy of {}".format(lreg_accuracy))

    lreg_model_coef = lreg_model.coef_
    for attribute, coefficient in zip(used_attribtues, lreg_model_coef):
        # Print the attributes and their respective coefficients
        print("Attribute: {}\t\t coefficient value: {}".format(attribute, coefficient))
    
    print("Linear Regression Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))  # The mean squared error
    print("Linear Regression Variance score: %.2f" % r2_score(y_test, y_pred))               # Explained variance score: 1 is perfect prediction

    plt.plot(y_test, color = "red")
    plt.plot(y_pred, color = "green")
    plt.title("Linear Regression Model (Training set)")
    plt.xlabel("x")
    plt.ylabel("Tensile Strength")
    # plt.show()

    print('\nK Nearest Neighbours Regression\n')
    
    ''' # KNN tests to find optimum number of enoughbours and distance type  
    # List of parameter to set power parameter for the Minkowski metric. 
    distance_type = [1,2] # p = 1 manhattan_distance, p = 2 euclidean_distance
    max_neigbours = 21
    for distance_pameter in distance_type:
        print(" Manhattan Distance") if distance_pameter == 1 else print("\n Euclidian Distance")
        for i in range(max_neigbours):
            num_neighbours = i + 1
            print("KNN Number of neighbours: {} -".format(num_neighbours), end = " ")
            knnReg = KNeighborsRegressor(n_neighbors=num_neighbours, p=distance_pameter)
            knn_reg_model = knnReg.fit(X_train, y_train) 
            knn_y_pred = knn_reg_model.predict(X_test) # Predcitions based off our testing data
            print("Mean squared error: %.2f, Variance score: %.2f" % (mean_squared_error(y_test, knn_y_pred), r2_score(y_test, knn_y_pred)))  # The mean squared error
    '''
    manhattan = 1
    n_neighbours = 13
    knnReg = KNeighborsRegressor(n_neighbors=n_neighbours, p=manhattan)

    knn_reg_model = knnReg.fit(X_train, y_train) 
    knn_reg_accuracy = knn_reg_model.score(X_test, y_test)  # Returns the coefficient of determination R^2 of the prediction.

    knn_y_pred = knn_reg_model.predict(X_test) # Predcitions based off our testing data
    print("KNN Regression model has an accuracy of {}".format(knn_reg_accuracy))

    print("KNN Regression Mean squared error: %.2f" % mean_squared_error(y_test, knn_y_pred))  # The mean squared error
    print("KNN Regression Variance score: %.2f" % r2_score(y_test, knn_y_pred))               # Explained variance score: 1 is perfect predictio

    kFold_validation(10, X, y)

# function written by Sean Harkin - Also used in Assignment 1 & 2
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

# function written by Sean Harkin - Also used in Assignment 1 & 2
def Average(list):
    return sum(list) / len(list)

# function written by Sean Harkin - Also used in Assignment 
def kFold_validation(n_splits, X_features, y_targets):

    lreg_times = []
    lreg_kfold_ratios = []
    knn_times = []
    knn_kfold_ratios = []

    '''10-fold cross validation'''

    # random_state=0 used when shuffle == True.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    no_splits = kf.get_n_splits(X_features)
    kfold_iteration = 1

    print("Number of splits: {}\n" .format(no_splits))

    for train_index, test_index in kf.split(X_features):
        print("K-Fold iteration number {}".format(kfold_iteration))
        # print("TRAIN:", train_index, "\nTEST:", test_index)

        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y_targets[train_index], y_targets[test_index]

        lreg = LinearRegression()
        lreg_start_time = time.time()
        lreg_model = lreg.fit(X_train, y_train) 
        lreg_times.append(time.time() - lreg_start_time)
        lreg_predicitons = lreg_model.predict(X_test)
        ratio = r2_score(y_test, lreg_predicitons)
        lreg_kfold_ratios.append(ratio)

        knnReg = KNeighborsRegressor(n_neighbors=13, p=1)
        knn_start_time = time.time()
        knn_reg_model = knnReg.fit(X_train, y_train)
        knn_times.append(time.time() - knn_start_time) 
        knn_predictions = knn_reg_model.predict(X_test)
        ratio = r2_score(y_test, knn_predictions)
        knn_kfold_ratios.append(ratio)

        kfold_iteration = kfold_iteration + 1
    
    lreg_average = Average(lreg_kfold_ratios)
    knn_average = Average(knn_kfold_ratios)

    lreg_average_time = Average(lreg_times)
    knn_average_time = Average(knn_times)


    print("\n10-fold Cross Validation Results:")
    print("Linear Regression average: {}, average training time: {}".format(lreg_average, lreg_average_time))
    print("KNN average: {}, average training time: {}".format(knn_average, knn_average_time))


if __name__ == '__main__':
    main()