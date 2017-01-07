
class myclass():
    def __init__(self):
        self.attr = None

    def compute_attribute(self):
        self.attr = 1

    def print_attribute(self):
        if self.attr is None:
            self.compute_attribute()
        print self.attr


class myclass2():
    def __init__(self):
        pass

    def compute_attribute(self):
        self.attr = 1
        return self.attr

    def print_attribute(self):
        try:
            attr = self.attr
        except AttributeError:
            attr = self.compute_attribute()
        if attr is not None:
            print attr


class myclass3():
    def __init__(self):
        self.attr = None
        pass

    @property
    def attr(self):
        if self.attr is None:
            self.attr = self.compute_attribute()
        return self.attr

    def compute_attribute(self):
        attr = 1
        return attr

    def print_attribute(self):
        print self.attr


class testA(object):

  def __init__(self):
    self.a = None
    self.b = None

  @property
  def a(self):
    if self.a is None:
      # Calculate the attribute now
      self.a = 7
    return self.a

  def myprint(self):
      print self.a

testclass = testA()
testclass.myprint()



def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


class MyClass(object):
    @lazyprop
    def attr(self):
        print('Generating attr')
        a =  self.generate()
        return a

    def generate(self):
        return 2

    @lazyprop
    def attr2(self):
        return 2+2

    def method(self):
        return self.attr

o = MyClass()
o.method()



class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value


class MyClass(object):
    @lazy_property
    def attr(self):
        print('Generating attr')
        a =  self.generate()
        return a

    def generate(self):
        return 2

    def method(self):
        return self.attr

    def method2(self):
        return self.attr

o = MyClass()
o.method()