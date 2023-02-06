def interface(path):
    """Initiates and displays widgets from iPyWigets
    and sends data to demo() with interactive_output()
    """

    import numpy as np
    import ipywidgets as widgets

    from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Label
    
    from read_file import read_file
    from demo import demo
    
    import warnings
    warnings.filterwarnings('ignore')
    
    interface.path = path


    t, C, T = read_file(path)
    
    if len(T.shape) != 0:
        cut = len(T) - 1
    else:
        cut = 1
    Index = widgets.IntSlider(
        value=0,
        min=0, # max exponent of base
        max=cut, # min exponent of base
        step=1, # exponent step
        description='')

    Methods = widgets.SelectMultiple(
        options = ['FISTA', 'L2', 'L1+L2', 'Contin', 'reSpect'],
        value   = ['Contin'],
        #rows    = 10,
        description = 'Methods:',
        disabled = False)

    Nz = widgets.IntText(
        value=100,
        description=r'N<sub>f</sub> =',
        disabled=False)

    Reg_L1 = widgets.FloatLogSlider(
        value=1e-8,
        base=10,
        min=-10, # max exponent of base
        max=1, # min exponent of base
        step=0.2, # exponent step
        description=r'FISTA: λ<sub>1</sub>')

    Reg_L2 = widgets.FloatLogSlider(
        value=1e-8,
        base=10,
        min=-10, # max exponent of base
        max=1, # min exponent of base
        step=0.2, # exponent step
        description=r'L2: λ<sub>2</sub>')
    
    Reg_C = widgets.FloatLogSlider(
        value=1E-1,
        base=10,
        min=-8, # max exponent of base
        max=2, # min exponent of base
        step=0.2, # exponent step
        description=r'Contin: λ<sub>C</sub>')
    
    Reg_S = widgets.FloatLogSlider(
        value=1E-2,
        base=10,
        min=-12, # max exponent of base
        max=2, # min exponent of base
        step=0.2, # exponent step
        description=r'reSpect: λ<sub>S</sub>')

    Bounds = widgets.IntRangeSlider(
        value=[-2, 2],
        min=-5,
        max=5,
        step=1,
        description=r'10<sup>a</sup> to 10<sup>b</sup>:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    
    dt = widgets.BoundedFloatText(
        value=150,
        min=0,
        max=1000,
        step=1,
        description='Time step, ms',
        disabled=False)

    Plot = widgets.ToggleButton(
        value=True,
        description='Hide graphics?',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Hides graphics',
        icon='eye-slash')
    
    Residuals = widgets.ToggleButton(
        value=False,
        description='Compute L-curve?',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots L-curve to choose best value of regularization parameter of L2 reg. method',
        icon='calculator')
    
    LCurve = widgets.Checkbox(
        value = False,
        description = 'Use L-Curve optimal?',
        disabled = False)
    
    Arrhenius = widgets.Checkbox(
        value = False,
        description = 'Draw Arrhenius instead DLTS?',
        disabled = False)

    Heatplot = widgets.ToggleButton(
        value=False,
        description='Plot heatmap?',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots heatmap of data from chosen file',
        icon='braille')


    left_box = VBox([Methods, Nz, dt])
    centre_box = VBox([Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds])
    right_box = VBox([LCurve, Arrhenius, Plot, Residuals, Heatplot])
    ui = widgets.HBox([left_box, centre_box, right_box])
    Slider = widgets.HBox([Label('Transient №'),Index])
    out = widgets.interactive_output(demo, {'Index':Index,   'Nz':Nz, 
                                            'Reg_L1':Reg_L1, 'Reg_L2':Reg_L2, 'Reg_C':Reg_C, 'Reg_S':Reg_S, 
                                            'Bounds':Bounds, 'dt':dt,         'Methods':Methods,
                                            'Plot':Plot,     'LCurve':LCurve, 'Arrhenius':Arrhenius,
                                            'Residuals':Residuals, 'Heatplot': Heatplot})
    display(ui, Slider, out)
