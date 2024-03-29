import os
import ipywidgets as widgets

l = widgets.Layout(width='50%')

class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()

    def _update_files(self):
        self.files = list()
        self.folders = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = os.path.join(self.path, f)
                if os.path.isdir(ff):
                    self.folders.append(f)
                else:
                    self.files.append(f)

    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box

    def _update(self, box):

        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update(box)

        buttons = []
        if self.files or self.folders:
            button = widgets.Button(layout = l, description='..', button_style='primary')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.folders:
            if f[0] != '.' and f[:2] != '__' and f != 'processed' and f != 'modules':
                button = widgets.Button(layout = l, description=f, button_style='info')
                button.on_click(on_click)
                buttons.append(button)
        for f in self.files:
            if f[0] != '.' and f[-5:] == '.DLTS' or f[-5:] == '.PERS':
                button = widgets.Button(layout = l, description=f, button_style='success')
                button.on_click(on_click)
                buttons.append(button)

        box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)
