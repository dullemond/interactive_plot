from interactive_plot import *

def viewarr(data,index=0,x=None,ymin=None,ymax=None,ylabel=None,    \
            idxnames=None,idxvals=None,idxformat='',iparstart=None, \
            fig=None, ax=None):
    """
    Interactive plot of a 1-D cut from an n-dimensional array.

    EXAMPLE 1:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data)

    EXAMPLE 2:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1)

    EXAMPLE 3:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1,idxnames=['ix','iy','iz'])

    EXAMPLE 4:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    viewarr(data,index=1,idxnames=['x','y','z'],idxvals=[['a','b','c','d'],[-3,-1,1,3],[1.0,2.0,3.0,4.0]])

    EXAMPLE 5:
    from viewarr import *
    data1=np.arange(64).reshape((4,4,4)) # Dummy dataset
    data2=64-data1
    viewarr([data1,data2],index=1,idxnames=['x','y','z'],idxvals=[['a','b','c','d'],[-3,-1,1,3],[1.0,2.0,3.0,4.0]],ylabel=['Bla','adfsd'])
    """
    import matplotlib.pyplot as plt
    if type(data)==list:
        shape =  data[0].shape
        ndim  = len(shape)
    else:
        shape = data.shape
        ndim  = len(shape)
    assert index<ndim, "Index out of range"
    idxorder  = list(range(ndim))
    idxorder.pop(index)
    idxorder.append(index)
    if type(data)==list:
        datatrans = []
        for d in data:
            datatrans.append(d.transpose(idxorder))
        shapetrans = datatrans[0].shape
    else:
        datatrans  = data.transpose(idxorder)
        shapetrans = datatrans.shape
    def func(x,param,fixedpar={"datatrans":datatrans}):
        datatrans = fixedpar["datatrans"]
        if type(datatrans)==list:
            answer = []
            for dslice in datatrans:
                for i in range(len(param)):
                    dslice = dslice[param[i]]
                answer.append(dslice)
        else:
            dslice = datatrans
            for i in range(len(param)):
                dslice = dslice[param[i]]
            answer = dslice
        answer = np.array(answer)
        return answer
    params=[]
    for i in range(ndim-1):
        params.append(np.arange(shapetrans[i]))
    if x is None:
        if idxvals is None:
            x = np.arange(shapetrans[-1])
        else:
            x = np.array(idxvals[index])
    if ymin is None:
        if type(data)==list:
            ymin = []
            for d in data:
                ymin.append(d.min())
            ymin = np.array(ymin).min()
        else:
            ymin = data.min()
    if ymax is None:
        if type(data)==list:
            ymax = []
            for d in data:
                ymax.append(d.max())
            ymax = np.array(ymax).max()
        else:
            ymax = data.max()
    if idxvals is not None:
        paramsalt = []
        for i in range(ndim-1):
            paramsalt.append(idxvals[idxorder[i]])
    else:
        paramsalt = None
    if idxnames is None:
        parnames = []
        for i in range(ndim-1):
            s = 'Parameter {}'.format(idxorder[i])
            parnames.append(s)
        xname    = 'Parameter {}'.format(index)
    else:
        parnames = []
        for i in range(ndim-1):
            parnames.append(idxnames[idxorder[i]]+" =")
        xname    = idxnames[index]
    if fig is None: fig = plt.figure()
    if ax is None:  ax  = plt.axes(xlim=(x.min(),x.max()),ylim=(ymin,ymax))
    ax.set_xlabel(xname)
    if ylabel is not None:
        if type(ylabel)==list:
            label = r''
            glue  = ''
            for l in ylabel:
                label += glue+l
                glue = ', '
            ax.set_ylabel(label)
        else:
            ax.set_ylabel(ylabel)
    if type(data)==list:
        axmodel = []
        if ylabel is None:
            for i in range(len(datatrans)):
                axm0,  = ax.plot(x,x,label='{}'.format(i))
                axmodel.append(axm0)
        else:
            for i in range(len(datatrans)):
                axm0,  = ax.plot(x,x,label=ylabel[i])
                axmodel.append(axm0)
        ax.legend()
    else:
        axmodel = None
    interactive_plot(x, func, params, ymin=ymin, ymax=ymax, parnames=parnames, parunits=None, fig=fig, ax=ax, axmodel=axmodel, parstart=None, iparstart=iparstart, plotbutton=False, fixedpar=None, returnipar=False, block=False, paramsalt=paramsalt, altformat=idxformat)


def slicearr(data,indices=(0,1),x=None,y=None,zmin=None,zmax=None,idxnames=None,idxvals=None,idxformat=''):
    """
    Interactive plot of a 2-D slice from an n-dimensional array.

    EXAMPLE 1:
    from viewarr import *
    data=np.arange(64).reshape((4,4,4)) # Dummy dataset
    slicearr(data,indices=(0,1))
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import NonUniformImage
    from matplotlib import cm
    from matplotlib import colors
    def img_func(param,fixedpar={}):
        if fixedpar['npar']==1: image = fixedpar['f'][:,:,param[0]]
        if fixedpar['npar']==2: image = fixedpar['f'][:,:,param[0],param[1]]
        if fixedpar['npar']==3: image = fixedpar['f'][:,:,param[0],param[1],param[2]]
        if fixedpar['npar']==4: image = fixedpar['f'][:,:,param[0],param[1],param[2],param[3]]
        if fixedpar['npar']==5: image = fixedpar['f'][:,:,param[0],param[1],param[2],param[3],param[4]]
        return image
    npar = len(data.shape)-2
    assert npar>0, 'Array must be 3D or higher dimension.'
    nx = data.shape[indices[0]]
    ny = data.shape[indices[1]]
    paridx = list(np.arange(len(data.shape)))
    paridx.remove(indices[0])
    paridx.remove(indices[1])
    parsiz=[]
    for i in range(len(paridx)):
        parsiz.append(data.shape[paridx[i]])
    if x is None: x = np.arange(nx)
    if y is None: y = np.arange(ny)
    if zmin is None: zmin=data.min()
    if zmax is None: zmax=data.max()
    norm     = colors.Normalize(vmin=zmin,vmax=zmax)
    cmap     = cm.hot
    fig,ax   = plt.subplots()
    im       = NonUniformImage(ax,interpolation='nearest',cmap=cmap,norm=norm)
    idcs     = [indices[0],indices[1]]
    for i in range(len(paridx)):
        idcs.append(paridx[i])
    datatrns = np.transpose(data,idcs)
    fixedpar = {}
    fixedpar['f']=datatrns
    fixedpar['npar']=npar
    param = np.zeros(npar,dtype=int)
    image = img_func(param,fixedpar=fixedpar)
    im.set_data(x,y,image.T)
    ax.images.append(im)
    ax.set_xlim((x[0]-0.5*(x[1]-x[0]),x[-1]+0.5*(x[-1]-x[-2])))
    ax.set_ylim((y[0]-0.5*(y[1]-y[0]),y[-1]+0.5*(y[-1]-y[-2])))
    cbar=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax)
    #cbar.set_label(r'$T\;[\mathrm{K}]$')
    params = []
    for i in range(len(paridx)):
        params.append(np.arange(parsiz[i])) # Choices of parameter values
    interactive_plot(None, None, params, fixedpar=fixedpar,       \
                     img_x=x,img_y=y,img_func=img_func,img_im=im, \
                     fig=fig,ax=ax)
