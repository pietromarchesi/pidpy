import numpy as np

def conditional_probability_from_joint(joint):
    X_mar = joint.sum(axis = 1)
    y_mar = joint.sum(axis = 0)

    cond_Xy = joint.astype(float) / y_mar[np.newaxis, :]
    cond_yX = joint.astype(float) / X_mar[:, np.newaxis]
    return cond_Xy, cond_yX


def specific_info(label, joint):
    y_mar_ = joint.sum(axis = 0)
    cond_Xy, cond_yX = conditional_probability_from_joint(joint)

    Ispec = 0
    for x in xrange(cond_Xy.shape[0]):
        contrib = cond_Xy[x, label] * (np.log2(1.0 / y_mar_[label])
                                       - np.log2(1.0 / cond_yX[x, label]))
        if np.isnan(contrib):
            contrib = 0
        Ispec += contrib
    return Ispec


def spec_info_full(self, labels, joints):
    spec_info_full_ = []
    for lab in labels:
        spec_info_lab = []
        for joint in joints:
            info = self.specific_info(lab, joint)
            spec_info_lab.append(info)
        spec_info_full_.append(spec_info_lab)
    return spec_info_full_


from pidpy import PIDCalculator
from pidpy.utilsc import _specific_info

X = np.random.randint(2,size = [10000,2])
y = np.random.randint(5,size = 10000)

pid = PIDCalculator(X,y)
joint = pid.joint_full_

lab = 4
si = specific_info(lab,joint)

cond_Xy, cond_yX = conditional_probability_from_joint(joint)
sic = _specific_info(lab, joint.sum(axis = 0),
                     cond_Xy, cond_yX, joint)

assert(si == sic)