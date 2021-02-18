import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QCalendarWidget, QComboBox,QLineEdit, QPushButton,  QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QDate
import torch
from model import CNN
from Literature_model import SJ_net
from datetime import datetime
import pandas_datareader.data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    argument = sys.argv
    if len(argument) == 1:
        model_name = "CNN_many"
    else:
        model_name = argument[1]
        if model_name == "CNN":
            model_name = "CNN_many"
            print("CNN model")
        elif model_name == "Combined":
            model_name = "Literature_net"
            print("Combined model")
        else:
            print("Not a valid model name. Use Default CNN model")
            model_name = "CNN_many"

class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        # Generate a calendar object
        self.cal = QCalendarWidget(self)
        self.cal.setGridVisible(True)
        self.cal.clicked[QDate].connect(self.showDate)

        # Generate a label which displays the selected date and indicates the stock name
        self.label_date = QLabel(self)
        self.todays_date = QLabel('Date: ', self)
        self.label_date.setText( self.cal.selectedDate().toString('yyyy-MM-dd'))
        #################################################
       # self.label_stock_name = QLabel('$Stock Name$: ', self)

        # Generate a label which displays the selected company
      #  self.label_company = QLabel('(SELECT THE STOCK FIRST)', self)
        self.label_1  = QLabel('Percentage of 1% raise for Korea_Elec: ', self)
        self.label_2 = QLabel('Percentage of 1% raise for Hyundai_Car: ', self)
        self.label_3 = QLabel('Percentage of 1% raise for SK_Tel: ', self)
        self.label_4 = QLabel('Percentage of 1% raise for SK_Hynix: ', self)
        self.label_5 = QLabel('Percentage of 1% raise for Samsung_Elec: ', self)
        self.label_6 = QLabel('Percentage of 1% raise for Posco: ', self)
        self.label_7 = QLabel('Percentage of 1% raise for KT&G: ', self)
        self.label_8 = QLabel('Percentage of 1% raise for Hyundai_Mobis: ', self)
        self.label_9 = QLabel('Percentage of 1% raise for Lotte_Chem: ', self)
        self.label_10 = QLabel('Percentage of 1% raise for S_Oil: ', self)
        self.label_11 = QLabel('Percentage of 1% raise for Shinhan_Fin: ', self)
        self.label_12 = QLabel('Percentage of 1% raise for Hanwha_Solution: ', self)
        self.label_13 = QLabel('Percentage of 1% raise for Korea_Shipbuilding: ', self)
        self.label_14 = QLabel('Percentage of 1% raise for Korea_Zinc: ', self)
        self.label_15 = QLabel('Percentage of 1% raise for Mirae_Asset: ', self)


        # Generate a black line which displays the favorable decision: Buy or not.
        self.answer_1 = QLineEdit(self)
        self.answer_2 = QLineEdit(self)
        self.answer_3 = QLineEdit(self)
        self.answer_4 = QLineEdit(self)
        self.answer_5 = QLineEdit(self)
        self.answer_6 = QLineEdit(self)
        self.answer_7 = QLineEdit(self)
        self.answer_8 = QLineEdit(self)
        self.answer_9 = QLineEdit(self)
        self.answer_10 = QLineEdit(self)
        self.answer_11 = QLineEdit(self)
        self.answer_12 = QLineEdit(self)
        self.answer_13 = QLineEdit(self)
        self.answer_14 = QLineEdit(self)
        self.answer_15 = QLineEdit(self)

        # Generate a black line which displays the best company to buy
       # self.best_company = QLineEdit(self)

        # Generate a black line which displays the inclination percent
       # self.percent_answer = QLineEdit(self)

        # Generate other 2 labels to complete the sentence: '
        #self.label_You_d_better = QLabel("You'd better", self)
        #self.label_percentage = QLabel('Percentage of 1% raise: ', self)
       # self.label_recommendation = QLabel('We recommend you to buy', self)

        # Generate a button which triggers the calculation
        self.button_go = QPushButton('GO!!!', self)
        self.button_go.setCheckable(False)
        self.button_go.clicked.connect(self.calculate)

        # Generate a button which clears the result
        self.button_clear = QPushButton('CLEAR.', self)
        self.button_clear.setCheckable(False)
        self.button_clear.clicked.connect(self.clear)

        # Generate a combobox object, and add selectable companies
       # cb = QComboBox(self)
        #cb.addItem('*CHOOSE STOCK*')
       # cb.addItem('Samsung_Elec')
       # cb.addItem('SK_Hynix')
       # cb.addItem('Naver')
       # cb.addItem('Kakao')

        # This changes the text of the label to the new selected company
       # cb.activated[str].connect(self.onActivated)

        # Place above items using hbox and vbox
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        hbox7 = QHBoxLayout()
        hbox8 = QHBoxLayout()
        hbox9 = QHBoxLayout()
        hbox10 = QHBoxLayout()
        hbox11 = QHBoxLayout()
        hbox12 = QHBoxLayout()
        hbox13 = QHBoxLayout()
        hbox14 = QHBoxLayout()
        hbox15 = QHBoxLayout()
        hbox16 = QHBoxLayout()
        hbox17 = QHBoxLayout()
        hbox18 = QHBoxLayout()
        hbox1.addWidget(self.cal)

        hbox2.addStretch(1)
        hbox2.addWidget(self.button_go)
        hbox2.addWidget(self.button_clear)
        hbox2.addStretch(1)

        hbox3.addStretch(1)
        hbox3.addWidget(self.label_1)
        hbox3.addWidget(self.answer_1)
        hbox3.addStretch(1)

        hbox4.addStretch(1)
        hbox4.addWidget(self.label_2)
        hbox4.addWidget(self.answer_2)
        hbox4.addStretch(1)

        hbox5.addStretch(1)
        hbox5.addWidget(self.label_3)
        hbox5.addWidget(self.answer_3)
        hbox5.addStretch(1)

        hbox6.addStretch(1)
        hbox6.addWidget(self.label_4)
        hbox6.addWidget(self.answer_4)
        hbox6.addStretch(1)

        hbox7.addStretch(1)
        hbox7.addWidget(self.label_5)
        hbox7.addWidget(self.answer_5)
        hbox7.addStretch(1)

        hbox8.addStretch(1)
        hbox8.addWidget(self.label_6)
        hbox8.addWidget(self.answer_6)
        hbox8.addStretch(1)

        hbox9.addStretch(1)
        hbox9.addWidget(self.label_7)
        hbox9.addWidget(self.answer_7)
        hbox9.addStretch(1)

        hbox10.addStretch(1)
        hbox10.addWidget(self.label_8)
        hbox10.addWidget(self.answer_8)
        hbox10.addStretch(1)

        hbox11.addStretch(1)
        hbox11.addWidget(self.label_9)
        hbox11.addWidget(self.answer_9)
        hbox11.addStretch(1)

        hbox12.addStretch(1)
        hbox12.addWidget(self.label_10)
        hbox12.addWidget(self.answer_10)
        hbox12.addStretch(1)

        hbox13.addStretch(1)
        hbox13.addWidget(self.label_11)
        hbox13.addWidget(self.answer_11)
        hbox13.addStretch(1)

        hbox14.addStretch(1)
        hbox14.addWidget(self.label_12)
        hbox14.addWidget(self.answer_12)
        hbox14.addStretch(1)

        hbox15.addStretch(1)
        hbox15.addWidget(self.label_13)
        hbox15.addWidget(self.answer_13)
        hbox15.addStretch(1)

        hbox16.addStretch(1)
        hbox16.addWidget(self.label_14)
        hbox16.addWidget(self.answer_14)
        hbox16.addStretch(1)

        hbox17.addStretch(1)
        hbox17.addWidget(self.label_15)
        hbox17.addWidget(self.answer_15)
        hbox17.addStretch(1)

        hbox18.addStretch(1)
        hbox18.addWidget(self.todays_date)
        hbox18.addWidget(self.label_date)
        hbox18.addStretch(1)

        vbox = QVBoxLayout()

        vbox.addLayout(hbox1)
        vbox.addStretch(2)
        vbox.addLayout(hbox2)
        vbox.addStretch(2)
        vbox.addLayout(hbox3)
        vbox.addStretch(2)
        vbox.addLayout(hbox4)
        vbox.addStretch(2)
        vbox.addLayout(hbox5)
        vbox.addStretch(2)
        vbox.addLayout(hbox6)
        vbox.addStretch(2)
        vbox.addLayout(hbox7)
        vbox.addStretch(2)
        vbox.addLayout(hbox8)
        vbox.addStretch(2)
        vbox.addLayout(hbox9)
        vbox.addStretch(2)
        vbox.addLayout(hbox10)
        vbox.addStretch(2)
        vbox.addLayout(hbox11)
        vbox.addStretch(2)
        vbox.addLayout(hbox12)
        vbox.addStretch(2)
        vbox.addLayout(hbox13)
        vbox.addStretch(2)
        vbox.addLayout(hbox14)
        vbox.addStretch(2)
        vbox.addLayout(hbox15)
        vbox.addStretch(2)
        vbox.addLayout(hbox16)
        vbox.addStretch(2)
        vbox.addLayout(hbox17)
        vbox.addStretch(2)
        vbox.addLayout(hbox18)
        self.setLayout(vbox)

        # Set window title, set the geometry, and display it
        self.setWindowTitle('$$$Stock Prediction Program$$$')
        self.setGeometry(300, 300, 700, 800)
        self.show()

    # Calculate the decision by using pre-trained model
    def calculate(self):
        stock_names = ['Korea_Elec', 'Hyndai_Car', 'SK_Tel', 'SK_Hynix', 'Samsung_Elec', 'Posco', 'KT&G', 'Hyundai_Mobis', 'Lotte_Chem', 'S_Oil', 'Shinhan_Fin', 'Hanwha_Solution', 'Korea_Shipbuilding', 'Korea_Zinc', 'Mirae_Asset' ]
        stock_codes = ['015760.KS', '005380.KS', '017670.KS', '000660.KS', '005930.KS', '005490.KS', '033780.KS', '012330.KS', '011170.KS', '010955.KS', '055550.KS', '009830.KS', '009540.KS', '010130.KS', '006800.KS' ]
        selected_date = self.cal.selectedDate().toString("yyyy/M/d")
        y, M, d = selected_date.split('/')
        y_start, y_end, M ,d = int(y)-1, int(y), int(M), int(d)
        start = datetime(y_start, M, d)
        end = datetime(y_end, M, d)
        num = 0
        percents = []
        for j in range(len(stock_names)):
            stock_data = pdr.DataReader(stock_codes[j], 'yahoo', start, end)
            truncated_stock = stock_data[-32:]
            data = truncated_stock[['High', 'Low', 'Open', 'Close', 'Volume']]
            scaler = MinMaxScaler()
            X = scaler.fit_transform(data)
            X = torch.from_numpy(X)
            if j==0:
                Y = X
            else:
                Y = torch.cat((Y,X),1)
        if model_name == 'CNN_many':
            model = CNN('SJ20_2', 5*len(stock_names), 2, 'relu', True)
        else:
            model = SJ_net('SJ20_1', 'SJ20_2', 5*len(stock_names), 'relu', True, 5*len(stock_names), 128, 2, 1, 1, 0.0, len(stock_names))

        if torch.cuda.is_available:
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_name))
        model.eval()
        Y = Y.transpose(0,1).float().unsqueeze(0)
        z = model(Y)
        z = z.detach().numpy()
        z = np.exp(z)[0,1,:]
        self.answer_1.setText(str(z[0]))
        self.answer_2.setText(str(z[1]))
        self.answer_3.setText(str(z[2]))
        self.answer_4.setText(str(z[3]))
        self.answer_5.setText(str(z[4]))
        self.answer_6.setText(str(z[5]))
        self.answer_7.setText(str(z[6]))
        self.answer_8.setText(str(z[7]))
        self.answer_9.setText(str(z[8]))
        self.answer_10.setText(str(z[9]))
        self.answer_11.setText(str(z[10]))
        self.answer_12.setText(str(z[11]))
        self.answer_13.setText(str(z[12]))
        self.answer_14.setText(str(z[13]))
        self.answer_15.setText(str(z[14]))


       # raise_percent = np.exp(Y[num][0][1])
       # if raise_percent > 0.5:
        #    answer="buy"
        #else:
        #    answer="not buy"

        #recommended_choice = stock_names[percents.index(max(percents))]
        #self.answer.setText(answer)
        #self.percent_answer.setText(str(raise_percent))
        #if max(percents) > 0.5:
        #    self.best_company.setText(recommended_choice)
        #else:
        #    self.best_company.setText("nothing")

    # Clear the answer lines
    def clear(self):
        self.answer_1.setText(" ")
        self.answer_2.setText(" ")
        self.answer_3.setText(" ")
        self.answer_4.setText(" ")
        self.answer_5.setText(" ")
        self.answer_6.setText(" ")
        self.answer_7.setText(" ")
        self.answer_8.setText(" ")
        self.answer_9.setText(" ")
        self.answer_10.setText(" ")
        self.answer_11.setText(" ")
        self.answer_12.setText(" ")
        self.answer_13.setText(" ")
        self.answer_14.setText(" ")
        self.answer_15.setText(" ")

    # Additional functions
    def showDate(self):
        self.label_date.setText(self.cal.selectedDate().toString('yyyy-MM-dd'))

    def onActivated(self, text):
        if text != '*CHOOSE STOCK*':
            self.label_company.setText("["+text+"]")
        else:
            self.label_company.setText('(SELECT THE STOCK FIRST)')
            self.clear()
        self.label_company.adjustSize()


# Exit
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

