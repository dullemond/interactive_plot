# Interactive Plot and Viewarr

### Purpose

The Python script interactive_plot.py allows you to quickly create a Graphical
User Interface to a figure plotting 1-D function, given a set of parameters. The
parameters are then represented as sliders below the figure, and you can then
see how the function changes as a function of the parameters.

The function can be anything, even the outcome of a complicated model. As
long as you can package your model into a Python function, with a 1-D
coordinate x as input, as well as one or more parameters (say, a, b and c),
and one or more values as output.

The purpose of interactive_plot.py is to make it easier to investigate how the
results of simple (= quick-to-calculate) models are dependent on the parameters.

As an add-on to interactive_plot.py this package also contain the script
viewarr.py, which allows you to very quickly plot 1-D cuts through an
N-dimensional numpy array, scanning the other dimensions with sliders. It can be
helpful to get a better insight into the data in a complex high-dimensional
array. 

### Examples of use of interactive_plot.py

#### Example 1 (a simple function with one parameter):

    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '])

#### Example 1-a (As above, but now with a plotting button instead of automatic replot; useful for heavier models):

    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '],plotbutton=True)

#### EXAMPLE 1-b (Plotting the content of a pre-calculated 2-D array):

    from interactive_plot import *
    x       = np.linspace(0,2*np.pi,100)
    y_array = np.zeros((30,100))
    omega   = np.linspace(1,3.,30)
    for i in range(30): y_array[i,:] = np.sin(omega[i]*x)
    def func(x,param): return y_array[param[0],:]
    params  = [np.arange(30)] # Choices of parameter values
    interactive_plot(x, func, params)

#### EXAMPLE 2 (Model fitting to data):

    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x        = np.linspace(0,2*np.pi,100)
    data     = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    fig      = plt.figure(1)
    ax       = plt.axes(xlim=(x.min(),x.max()),ylim=(-1.2,1.2))
    axd,     = ax.plot(x,data,'o',label='data')
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params   = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    parstart = [0.6,2.0]  # Initial guesses for parameters
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, label='model',parstart=parstart)
    ax.legend()
    plt.show()

#### EXAMPLE 2-a (Model overplotting over an image):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x        = np.linspace(0,2*np.pi,100)
    image    = np.random.normal(size=(70,70)) # Make some image
    fig      = plt.figure(1)
    extent   = [x.min(),x.max(),-1.2,1.2]
    axd      = plt.imshow(image,extent=extent,cmap=cm.hot)
    ax       = plt.gca()
    plt.axis(extent)
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params   = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    parstart = [0.6,2.0]  # Initial guesses for parameters
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, label='model',parstart=parstart)
    ax.legend()
    plt.show()

#### EXAMPLE 3 (Fitting two models simultaneously to data):

    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return np.vstack((param[0]*np.sin(param[1]*x),param[0]*np.cos(param[1]*x)))
    x      = np.linspace(0,2*np.pi,100)
    data   = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    fig    = plt.figure(1)
    ax     = plt.axes(xlim=(x.min(),x.max()),ylim=(-1.2,1.2))
    axd,   = ax.plot(x,data,'o',label='data')
    axm0,  = ax.plot(x,data,'--',label='sin')
    axm1,  = ax.plot(x,data,':',label='cos')
    axmodel= [axm0,axm1]
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)]
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, axmodel=axmodel)
    ax.legend()
    plt.show()

#### EXAMPLE 3-a (Fitting two models in two separate plots simultaneously):

    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return np.vstack((param[0]*np.sin(param[1]*x),param[0]*np.cos(param[1]*x)))
    x         = np.linspace(0,2*np.pi,100)
    data      = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    extent    = [x.min(),x.max(),-1.2,1.2]
    fig, axes = plt.subplots(ncols=2)
    axes[0].axis(extent)
    axes[1].axis(extent)
    axd0,  = axes[0].plot(x,data,'o',label='data')
    axm0,  = axes[0].plot(x,data,'--',label='sin')
    axd1,  = axes[1].plot(x,data,'o',label='data')
    axm1,  = axes[1].plot(x,data,':',label='cos')
    axmodel= [axm0,axm1]
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)]
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=0, axmodel=axmodel)
    plt.show()

#### EXAMPLE 4: (passing additional fixed parameters to function):

    from interactive_plot import *
    def func(x,param,fixedpar={}): return param[0]*np.sin(param[1]*x)+fixedpar['offset']
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '],fixedpar={'offset':0.6})

#### EXAMPLE 5: (Interactive image, e.g. 2D slice from a higher-dimensional data box):

    import numpy as np
    from interactive_plot import *
    from matplotlib import cm
    from matplotlib import colors
    import matplotlib.pyplot as plt
    from matplotlib.image import NonUniformImage
    x        = np.linspace(-1,1,20)
    y        = np.linspace(-1,1,30)
    z        = np.linspace(0,1,25)
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    rr       = np.sqrt(xx**2+yy**2)
    f        = np.sin(xx*2*np.pi)*yy*(1-zz)+np.cos(2*np.pi*rr)*zz
    norm     = colors.Normalize(vmin=f.min(),vmax=f.max())
    cmap     = cm.hot
    fig,ax   = plt.subplots()
    im       = NonUniformImage(ax,interpolation='nearest',cmap=cmap,norm=norm)
    im.set_data(x,y,f[:,:,0].T)
    ax.images.append(im)
    ax.set_xlim((x[0]-0.5*(x[1]-x[0]),x[-1]+0.5*(x[-1]-x[-2])))
    ax.set_ylim((y[0]-0.5*(y[1]-y[0]),y[-1]+0.5*(y[-1]-y[-2])))
    cbar=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax)
    cbar.set_label(r'$T\;[\mathrm{K}]$')
    def img_func(param,fixedpar={}): return fixedpar['f'][:,:,param[0]]
    params = [np.arange(25)] # Choices of parameter values
    fixedpar = {}
    fixedpar["f"]=f
    interactive_plot(None, None, params, fixedpar=fixedpar,       \
                     img_x=x,img_y=y,img_func=img_func,img_im=im, \
                     fig=fig,ax=ax)

### Examples of use of viewarr.py

#### EXAMPLE 1:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data)

#### EXAMPLE 2:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1)

#### EXAMPLE 3:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1,idxnames=['ix','iy','iz'])

#### EXAMPLE 4:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1,idxnames=['x','y','z'],idxvals=[['a','b','c','d'],[-3,-1,1,3],[1.0,2.0,3.0,4.0]])

#### EXAMPLE 5:
    from viewarr import *
    data1=np.arange(64).reshape((4,4,4)) # Dummy dataset
    data2=64-data1
    viewarr([data1,data2],index=1,idxnames=['x','y','z'],idxvals=[['a','b','c','d'],[-3,-1,1,3],[1.0,2.0,3.0,4.0]],ylabel=['Bla','adfsd'])

### Package dependencies

`numpy`, `matplotlib`
