# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:05:22 2014

@author: rkmaddox
"""

import pyglet
import numpy as np
from pyglet.window import key


#==============================================================================
# Make this a classy party
#==============================================================================
class Maskers(object):
    def __init__(self, files=['maskers/NW_F_C.wav',
                              'maskers/NW_F_S.wav',
                              'maskers/NW_M_C.wav',
                              'maskers/NW_M_S.wav']):
        self.files = files
        self.sources = [pyglet.media.load(f, streaming=False)
                        for f in self.files]
        self.duration = np.max([s.duration for s in self.sources])
        self.players = [pyglet.media.Player()
                        for _ in range(len(self.files))]
        self.mask_vol = 1.0
        self.location = 0
        self.gender = 0
        self.set_vols()

    def loop(self, dt=None):
        for player, source in zip(self.players, self.sources):
            player.eos_action = 'loop'
            player.queue(source)
        for player in self.players:
            player.play()
        self.set_vols()

    def stop(self):
        for player in self.players:
            player.next()
            player.pause()

    def set_gender(self, gender):
        self.gender = gender
        self.set_vols()

    def set_location(self, location):
        self.location = location
        self.set_vols()

    def set_vols(self):
        self.play_ind = 2*self.gender + self.location
        self.vols = [0 for _ in range(len(self.files))]
        self.vols[self.play_ind] = self.mask_vol
        for p, v in zip(self.players, self.vols):
            p.volume = v
        print([int(v > 0) for v in [p.volume for p in self.players]])


class Target(object):
    def __init__(self, screen, file_name='targets/kix.mpeg',
                 max_vol=1.9952623149688795):
        self._screen = screen
        self.file = file_name
        self.source = pyglet.media.load(self.file, streaming=True)
        self.player = pyglet.media.Player()
        self.duration = self.source.duration
        self.player.volume = 1.
        self.max_vol = max_vol
        self.show = False

    def loop(self, dt=None):
        self.player.eos_action = 'loop'
        self.player.queue(self.source)
        self.player.play()
        self.player.volume = self.player.volume

        tex = self.player.get_texture()
        tex.anchor_x = int(tex.width / 2)
        tex.anchor_y = int(tex.height / 2)

        self.label_str = 'Level: %i'
        self.label = None
        self.update_label()

    def stop(self):
        self.player.next()
        self.player.pause()

    def set_volume(self, vol):
        self.player.volume = np.minimum(vol, self.max_vol)
        print(np.round(20 * np.log10(self.player.volume), 1))
        self.update_label()

    def get_volume(self):
        return self.player.volume

    def get_volume_db(self):
        return 20 * np.log10(self.get_volume())

    def update_label(self):
        if self.label is None:
            x = self._screen.width / 2
            y = self._screen.height / 2 - self.player.get_texture().height / 2
            self.label = pyglet.text.Label(text='X', x=x, y=y,
                                           anchor_x='center',
                                           anchor_y='top')
        self.label.text = self.label_str % np.round(self.get_volume_db() + 25)

    def draw(self):
        if self.show:
            self.player.get_texture().blit(screen.width / 2, screen.height / 2)
        self.label.draw()


#==============================================================================
# Initialize everything
#==============================================================================
# Set up the window and event handlers
display = pyglet.canvas.get_display()
screen = display.get_default_screen()
win_kwargs = dict(width=screen.width, height=screen.height,
                  caption='Maskers', fullscreen=True,
                  screen=0, style='borderless', visible=True)
window = pyglet.window.Window(**win_kwargs)
keys = key.KeyStateHandler()
window.push_handlers(keys)

# Get the party started
maskers = Maskers()
target = Target(screen)

# Set some parameters
vol_inc = 10. ** (1./20)
key_vis = key._1
key_location = key._2
key_gender = key._3


#==============================================================================
# Begin callback functrions
#==============================================================================
@window.event
def on_key_press(symbol, modifiers):
    if symbol == key_location:
        maskers.set_location(1)
    elif symbol == key_gender:
        maskers.set_gender(1)
    elif symbol == key_vis:
        target.show = True
    elif symbol == key.UP:
        target.set_volume(target.get_volume() * vol_inc)
    elif symbol == key.DOWN:
        target.set_volume(target.get_volume() / vol_inc)
    elif symbol == key.ESCAPE:
        target.stop()
        maskers.stop()
        window.has_exit = True


@window.event
def on_key_release(symbol, modifiers):
    if symbol == key_location:
        maskers.set_location(0)
    if symbol == key_gender:
        maskers.set_gender(0)
    elif symbol == key_vis:
        target.show = False


@window.event
def on_draw():
    window.clear()
    target.draw()


#==============================================================================
# Finish up and get it going
#==============================================================================
# Keep the party going
pyglet.clock.schedule_once(maskers.loop, 0)
pyglet.clock.schedule_once(target.loop, 0)

pyglet.app.run()
