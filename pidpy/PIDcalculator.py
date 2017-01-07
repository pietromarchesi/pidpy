def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class PIDCalculator():

    def __init__(self):
        pass

    def initialise(self,X,y):
        self.X = X
        self.y = y
        #self.joint_var_  = None
        #self.joint_sub_  = None
        #self.joint_full_ = None
        #self.spec_info_var_ = None
        #self.spec_info_sub_ = None
        #self.y_mar_ = None
        #self.mi_full = None

    @lazy_property
    def joint_var_(self):
        joint_var_ = #compute joint_var
        return joint_var_

    @lazy_property
    def spec_info_var_(self):
        spec_info_var_ = #compute spec_info_var_
        return spec_info_var_

    @lazy_property
    def y_mar_(self):
        y_mar_ = #compute y_mar_
        return y_mar_


    def redundancy(self):

        self.red = Imin(self.y_mar_, self.spec_info_var_)
        return self.red






def Imin(y_mar_, spec_info):
    Im = 0
    for i in len(y_mar_):
        Im += y_mar_[i]*min(spec_info[i])
    return Im