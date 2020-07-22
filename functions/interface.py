def interface():

    '''Initiates widgets'''

    from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Label
    import ipywidgets as widgets
    import numpy as np
    from demo import demo
    from read_file import read_file

    t, C, T = read_file('/Users/antonvasilev/GitHub/ilt/data/beta/EUNB29b_1-16-2_15_2.DLTS')

    cut = len(T) - 1
    Index = widgets.IntSlider(
        value=1,
        min=0, # max exponent of base
        max=cut, # min exponent of base
        step=1, # exponent step
        description='')

    Methods = widgets.SelectMultiple(
        options = ['L1', 'L2', 'L1+L2', 'SVD'],
        value   = ['SVD'],
        #rows    = 10,
        description = 'Methods:',
        disabled = False)

    Nz = widgets.IntText(
        value=100,
        description=r'$N_f=$',
        disabled=False)

    Reg_L1 = widgets.FloatLogSlider(
        value=0.1,
        base=10,
        min=-5, # max exponent of base
        max=0.5, # min exponent of base
        step=0.2, # exponent step
        description=r'L1: $\lambda_1$')

    Reg_L2 = widgets.FloatLogSlider(
        value=1E-3,
        base=10,
        min=-12, # max exponent of base
        max=-2, # min exponent of base
        step=0.2, # exponent step
        description=r'L2: $\lambda_2$')

    Bounds = widgets.IntRangeSlider(
        value=[-3, 2],
        min=-4,
        max=4,
        step=1,
        description=r'$10^{a}\div 10^{b}$:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')


    Plot = widgets.Checkbox(
        value = True,
        description = 'Plot?',
        disabled = False)

    Residuals = widgets.ToggleButton(
        value=False,
        description='Plot L-curve',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots L-curve to choose best value of regularization parameter of L2 reg. method',
        icon='plus')

    Heatplot = widgets.ToggleButton(
        value=False,
        description='Plot heatplot',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Plots heatmap of data from choosed file',
        icon='plus')


    left_box = VBox([Methods])
    centre_box = VBox([Nz, Reg_L1, Reg_L2, Bounds])
    right_box = VBox([Plot, Residuals, Heatplot])
    ui = widgets.HBox([left_box, centre_box, right_box])
    Slider = widgets.HBox([Label('Transient №'),Index])
    out = widgets.interactive_output(demo, {'Index':Index,
                                            'Nz':Nz, 'Reg_L1':Reg_L1, 'Reg_L2':Reg_L2,
                                            'Bounds':Bounds, 'Methods':Methods,
                                            'Plot':Plot, 'Residuals':Residuals,
                                            'Heatplot': Heatplot})
    display(ui, Slider, out)
