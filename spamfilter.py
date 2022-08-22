import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"),delimiter=",").astype(np.int32)

testing_spam = np.loadtxt(open("data/testing_spam.csv"),delimiter=",").astype(np.int32)

batch1 =training_spam[:100,1:]
batch1_value =training_spam[:100,:1]

batch2 =training_spam[100:200,1:]
batch2_value =training_spam[100:200,:1]

batch3 =training_spam[200:300,1:]
batch3_value =training_spam[200:300,:1]

batch4 =training_spam[300:400,1:]
batch4_value =training_spam[300:400,:1]

batch5 =training_spam[400:500,1:]
batch5_value =training_spam[400:500,:1]

batch6 =training_spam[500:600,1:]
batch6_value =training_spam[500:600,:1]

batch7 =training_spam[600:700,1:]
batch7_value =training_spam[600:700,:1]

batch8 =training_spam[700:800,1:]
batch8_value =training_spam[700:800,:1]

batch9 =training_spam[800:900,1:]
batch9_value =training_spam[800:900,:1]

batch10 =training_spam[900:1000,1:]
batch10_value =training_spam[900:1000,:1]

sample_set = np.array([batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10])
true_value_set = np.array([batch1_value,batch2_value,batch3_value,batch4_value,batch5_value,batch6_value,batch7_value,batch8_value,batch9_value,batch10_value])


class SpamClassifier:
    def __init__(self):
        self.learning_rate = 0.1
        self.input_layer =54
        self.hidden_layer = 20

        self.w1 = np.random.uniform(-1,1, (self.input_layer ,self.hidden_layer))

        self.w2 = np.random.uniform(-1,1,(self.hidden_layer ,1))
       
        self.b1 = np.random.uniform(-1,1,(self.hidden_layer ,1))
        
        self.b2 = np.random.uniform(-1,1,(1 ,1))

    def sigmoid(self,layer_input):
        return 1.0 / (1.0 + np.exp(-layer_input))
    
    def derivative_sig(self,sigmoid_function):
        return sigmoid_function * (1.0-sigmoid_function)


    def forward_prop(self,input_data):
        Z1 = input_data.dot(self.w1) + np.transpose(self.b1)
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.w2)+ self.b2
        A2 = self.sigmoid(Z2)

        return Z1, A1, Z2, A2
    
    def backward_prop(self,Z1, A1,A2, input_data, true_value):
        m = A2.size
        dZ2 = A2 - true_value
        dW2 = (1 / m )* A1.T.dot(dZ2)
        db2 = (1 / m) * np.sum(dZ2)
        dZ1 = (self.w2)*((self.derivative_sig(self.sigmoid((Z1)))).T.dot(dZ2))
        dW1 = (input_data.T).dot(((1 / m)*(self.w2).T)*((self.derivative_sig(self.sigmoid((Z1))))*(dZ2)))
        db1 = (1 / m)* np.sum(dZ1)
        return dW1, db1, dW2, db2

    def updates(self,dW1, db1, dW2, db2):
        step_size_bs = db2*(self.learning_rate)
        self.b2 -=step_size_bs

        step_size_ws = dW2*(self.learning_rate)
        self.w2 -=step_size_ws

        step_size_bf = db1*(self.learning_rate)
        self.b1 -=step_size_bf

        step_size_wf = dW1*(self.learning_rate)
        self.w1 -=step_size_wf

    def get_prediction(self,A2):
        for i in range(A2.size):
            if A2[i,0]>0.5:
                A2[i,0] = 1
            else:
                A2[i,0] = 0
        return A2

    def get_accuracy(self,j,predictions):
        return np.sum(predictions==j)/j.size

    def gradient_descent(self,sample_set,true_value_set):
        for number in range(100):
            for i,j in zip(sample_set,true_value_set):
                Z1, A1, Z2, A2 = self.forward_prop(i)
                dW1, db1, dW2, db2 = self.backward_prop(Z1, A1,A2,i,j)
                self.updates(dW1, db1, dW2, db2)  
                predictions = self.get_prediction(A2)
                accuracy= self.get_accuracy(j,predictions)
                print('Improving accuracy:',accuracy)

        return dW1, db1, dW2, db2

    def make_prediction(self,test_data):
        Z1, A1, Z2, A2 = self.forward_prop(test_data) 
        test_p = self.get_prediction(A2)
        return test_p

  

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int32)
test_data = testing_spam[:, 1:]
test_size =testing_spam[:, 0].size
test_result = testing_spam[:, 0].reshape(test_size,1)
test_labels =test_result


Classifier = SpamClassifier()
Classifier.gradient_descent(sample_set,true_value_set)
predictions = Classifier.make_prediction(test_data)

accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
print(f"Accuracy on test data is: {accuracy}")
        






