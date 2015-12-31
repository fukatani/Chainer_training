#-------------------------------------------------------------------------------
# Name:        deep_auto_encoder
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#%matplotlib inline
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import numpy as np
from auto_encoder import Autoencoder

#constructing newral network
class Deepautoencoder(Autoencoder):
    def forward(self, x_data, y_data, train=True, answer=False):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)),  train=train)
        h3 = F.dropout(F.relu(self.model.l3(h2)),  train=train)
        h4 = F.dropout(F.relu(self.model.l4(h3)),  train=train)
        h5 = F.dropout(F.relu(self.model.l5(h4)),  train=train)
        h6 = F.dropout(F.relu(self.model.l6(h5)),  train=train)
        h7 = F.dropout(F.relu(self.model.l7(h6)),  train=train)
        h8 = F.dropout(F.relu(self.model.l8(h7)),  train=train)
        h9 = F.dropout(F.relu(self.model.l9(h8)),  train=train)
        h10 = F.dropout(self.model.l10(h9), train=train)
        y  = self.model.l11(h10)
        if answer:
            return y.data, F.mean_squared_error(y, t)
        else:
            return F.mean_squared_error(y, t)#, F.accuracy(y, t)

    def set_model(self, nobias):
        try:
            self.load_from_pickle()
        except (IOError, EOFError):
            self.model = FunctionSet(l1=F.Linear(self.input_matrix_size, self.n_units, nobias=nobias),
                            l2=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l3=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l4=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l5=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l6=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l7=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l8=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l9=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l10=F.Linear(self.n_units, self.n_units, nobias=nobias),
                            l11=F.Linear(self.n_units, self.output_matrix_size, nobias=nobias)
                            )

if __name__ == '__main__':
    #Deepautoencoder(train_size=98, n_epoch=50, n_units=250, same_sample=10, offset_cancel=True)
    Deepautoencoder(train_size=98,
            n_epoch=30,
            n_units=300,
            batch_size=1,
            same_sample=10,
            offset_cancel=True,
            is_clastering=False,
            input_data_size=300,
            split_mode='pp',
            #nobias=True
            )
