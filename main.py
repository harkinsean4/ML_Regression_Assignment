import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

    '''Classification'''

    print("There are {} samples".format(len(y)))
    train_size = round((2/3)*len(y))

    # random_state set to zero so we generate the same randomised X and y data every time we run the code
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)

    print('\nLinear Regression Classification\n')

    lreg = LinearRegression()

    reg_model = lreg.fit(X_train, y_train) 
    reg_accuracy = reg_model.score(X_test, y_test)

    # Predcitions based off our testing data
    y_pred = reg_model.predict(X_test)

    print("Linear Regression model has an accuracy of {}".format(reg_accuracy))

    # Print the attributes and their respective coefficients
    reg_model_coef = reg_model.coef_
    for attribute, coefficient in zip(used_attribtues, reg_model_coef):
        print("Attribute: {}\t\t coefficient value: {}".format(attribute, coefficient))
    
    # The mean squared error
    print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    plt.plot(y_test, color = "red")
    plt.plot(y_pred, color = "green")
    plt.title("Linear Regression Model (Training set)")
    plt.xlabel("x")
    plt.ylabel("Tensile Strength")
    # plt.show()


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