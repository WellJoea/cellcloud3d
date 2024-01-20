import os
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from cellcloud3d.plotting._colors import color_palette, vartype, ListedColormap, cmap1

def surfaces3d(meshlist=None, points = None, groupby =None,#mesh_colors=None, 
               grouporder = None,  window_size=None,
                edge_colors='#AAAAAA', colors='#AAAAAA', point_cmap=None,
                show_actor = False, actor_size = 15, startpos = 10, actor_space=5,
                color_off='grey', color_bg='grey',
                vmin = None, vmax = None,
                show_edges=True, opacity=0.05, edge_opacity=None,
                notebook=True, 
                # theme_name=None, 
                theme=None, 
                background_color='black', font_color='white',
                font_size =25, title_size =30,
                main_title_font =None, main_title_color='white', main_title_size=30,
                point_size=0.5, cpos=None, show_scalar_bar=True,
                framerate=12, view_up = None, viewup=(0, 0, 0),
                factor=2, n_points=120, step=0.1, line_width=None,
                shift=0, focus=None, quality=8, write_frames=True,
                show=False,title=None, view_isometric=False,
                render_points_as_spheres=True, 
                anti_aliasing=True, lighting=False, aa_type ='fxaa', multi_samples=None, all_renderers=True,
                rotate_y = None, rotate_z = None, rotate_x = None, vtkz_format='json',
                save=None, save_type=None, 
                args_cbar = {},
                pkargs= dict(lighting=False), 
                **mkargs):
    import pyvista as pv

    # if theme_name:
    #     pv.set_plot_theme(theme_name)
        # pv.set_plot_theme('default')
        # pv.global_theme

    # if theme is None:
    #     my_theme = pv.themes.Theme()
    #     # my_theme.background.color = background_color
    #     # my_theme.set_font(size=40, color='red')
    #     my_theme.font.color = 'red'
    #     my_theme.font.size = 40
    #     print(my_theme)
    #     pv.theme = my_theme

    if theme is None:
        theme = pv.themes.DarkTheme()
        # theme.lighting = True
        # theme.show_edges = True
        # theme.edge_color = 'white'
        # theme.color = 'black'
        theme.background = background_color
        theme.font.color = font_color
        theme.font.size  = font_size
        theme.font.title_size = title_size

    sclbar = dict(height=0.8, width=0.05, vertical=True,
                        interactive=False,
                        n_labels=5,
                        position_x=0.95, position_y=0.1, color="white", 
                        title_font_size=30, label_font_size=25)
    sclbar.update(args_cbar)

    pl = pv.Plotter(notebook=notebook, window_size=window_size, theme=theme)
    if meshlist:
        if isinstance(colors, str) or (colors is None):
            colors = [colors]*len(meshlist)
        if isinstance(edge_colors, str) or (edge_colors is None):
            edge_colors = [edge_colors]*len(meshlist)

        for j, mesh in enumerate(meshlist):
            pl.add_mesh(mesh, show_edges=show_edges, opacity=opacity, 
                        edge_opacity=edge_opacity, line_width=line_width,
                        edge_color=edge_colors[j], color=colors[j], lighting=lighting, **mkargs)

            if not rotate_x is None:
                mesh = mesh.rotate_x(rotate_x, inplace=False)
            if not rotate_y is None:
                mesh = mesh.rotate_y(rotate_y, inplace=False)
            if not rotate_z is None:
                mesh = mesh.rotate_z(rotate_z, inplace=False)

    if not points is None:
        if type(points) == pv.PolyData:
            pvdt = points.copy()
            try:
                groups = np.array(pvdt[groupby])
            except:
                groups = np.ones(pvdt.points.shape[0]).astype(str)
            gdtype = vartype(groups)
            if (gdtype == 'discrete'):
                Order = np.unique(groups) if grouporder is None else grouporder
                groups = pd.Categorical(groups, categories=Order)
                # reorder = dict(zip( Order, np.arange(len(Order)) ))
                # groups = groups.rename_categories(reorder).astype(np.int64)
                # annotations =  {v:k for k,v in reorder.items()}
                pvdt[groupby] = groups

        elif type(points) in [np.ndarray, pd.DataFrame]:
            pvdt = pv.PolyData(np.array(points)[:,[0,1,2]].astype(np.float64))
            if groupby is None:
                groups = np.ones(points.shape[0]).astype(str)
            else:
                groups = points[groupby]
                pvdt[groupby] = groups
            gdtype = vartype(groups)
        
        icolor = None
        if gdtype == 'discrete':
            try:
                Order = groups.cat.remove_unused_categories().cat.categories
            except:
                Order= np.unique(groups)
            if point_cmap is None:
                color_list = color_palette(len(Order))
            else:
                color_list = [point_cmap] if isinstance(point_cmap, str) else point_cmap
                if len(color_list)  == 1:
                    icolor = color_list[0]

            # TODO
            coldict = dict(zip(Order, color_list))
            Order = np.sort(Order)
            color_list = [ coldict[i] for i in Order ]
            my_cmap = ListedColormap(color_list)

            categories = True
            sclbar["n_labels"] = 0 #len(Order)
            sclbar["n_colors"] = len(Order)
            # sclbar["height"] =  0.85
  
        elif gdtype == 'continuous':
            try:
                pvdt.set_active_scalars(groupby)
                if vmin is None:
                    vmin = pvdt[groupby].min()
                if vmax is None:
                    vmax = pvdt[groupby].max()
                pvdt = pvdt.threshold([vmin, vmax])
            except:
                pass
            show_actor = False
            categories = False
            my_cmap = cmap1 if point_cmap is None else point_cmap

        if (not show_actor):
            pl.add_mesh(pvdt, 
                        style='points',
                        cmap=my_cmap,
                        color=icolor, 
                        label = None if groupby is None else str(groupby),
                        scalars = groupby,
                        scalar_bar_args=sclbar,
                        render_points_as_spheres=render_points_as_spheres, 
                        show_scalar_bar=show_scalar_bar,
                        point_size=point_size,
                        categories=categories,
                        # annotations=annotations,
                        **pkargs)
        elif show_actor and (gdtype == 'discrete'):
            Startpos = startpos
            for i, iorder in enumerate(Order):
                idx = groups == iorder
                ipos = pvdt.points[idx]
                icolor = color_list[i]

                ipos = pv.PolyData(ipos)
                actor = pl.add_mesh(ipos, style='points', 
                                    cmap=my_cmap,
                                    label = str(iorder), #groupby,
                                    name = str(iorder),
                                    color=icolor, 
                                    render_points_as_spheres=render_points_as_spheres,
                                    # scalars = scalars,
                                    # scalar_bar_args=sclbar,
                                    # label=str(i),
                                    categories=categories,
                                    point_size=point_size)

                callback = SetVisibilityCallback(actor)
                pl.add_checkbox_button_widget(
                    callback,
                    value=True,
                    position=(5.0, Startpos),
                    size=actor_size,
                    border_size=1,
                    color_on=icolor,
                    color_off=color_off,
                    background_color=color_bg,
                )
                # pl.add_actor(actor, reset_camera=False, name=str(i), 
                #              culling=False, pickable=True, 
                #              render=True, remove_existing_actor=True)
                Startpos = Startpos + actor_size + (actor_size // actor_space)
            # pl.add_legend(loc='center right', bcolor=None, size=[0.1,0.8], face='circle')
        # pl.update_scalar_bar_range([vmin, vmax], name=None)

    if anti_aliasing is None:
        pass
    elif anti_aliasing is True:
        pl.enable_anti_aliasing(aa_type, multi_samples=multi_samples, all_renderers=all_renderers)
    elif anti_aliasing is False:
        pl.disable_anti_aliasing()

    # add_legend(pl)

    if cpos:
        pl.camera_position = cpos
    if view_isometric:
        pl.view_isometric()
    if title:
        actor = pl.add_title(title, font=main_title_font, color=main_title_color, font_size=main_title_size)
    # pv.theme.restore_defaults()
    return save_mesh(save_type=None)(pl, save, show=show,
                    framerate=framerate, view_up = view_up, 
                    viewup=viewup, factor=factor, n_points=n_points, 
                    step=step, shift=shift, focus=focus,
                    quality=quality, write_frames=write_frames)

def surface(adata, meshlist=None, use_raw=False, groupby = None, splitby=None,
            basis='spatial3d', gcolors = None, outpre =None, format ='mp4',
            save=None,
            **kargs):
    import pyvista as pv
    pls = []
    if not splitby is None:
        if (type(splitby) == str):
            splitby = [splitby]
        for isplit in splitby:
            try:
                Order = adata.obs[isplit].cat.remove_unused_categories().cat.categories
            except:
                Order= adata.obs[isplit].unique()
            if f'{isplit}_colors' in adata.uns.keys():
                my_colors = adata.uns[f'{isplit}_colors']
            else:
                my_colors = color_palette(len(Order))

            for i, (icolor, iorder) in enumerate(zip(my_colors, Order)):
                idata = adata[adata.obs[isplit] == iorder]
                posdf = pd.DataFrame(idata.obsm[basis], columns=['x','y','z'])
                if not save is None:
                    save = save
                elif outpre:
                    save = f'{outpre}.{iorder}.{format}'
                else:
                    save = outpre
                ipl = surfaces3d(meshlist = meshlist, 
                                points=posdf, 
                                groupby=None, 
                                title=f'{iorder}',
                                point_cmap = icolor,
                                save = save,
                                **kargs)
                pls.append(ipl)
    else:
        if (groupby is None) or (type(groupby) == str):
            groupby = [groupby]
        for group in groupby:
            if group in adata.obs.columns:
                gdata = adata.obs[group].reset_index(drop=True)
                if not gcolors is None:
                    my_cmap = gcolors[group]
                elif f'{group}_colors' in adata.uns.keys():
                    my_cmap = adata.uns[f'{group}_colors']
                else:
                    my_cmap = color_palette(np.unique(gdata).shape[0])

            elif group in adata.var_names:
                if use_raw:
                    gdata = adata.raw.to_adata()[:, group].X
                else:
                    gdata = adata[:, group].X
                if issparse(gdata):
                    gdata = gdata.toarray().flatten()
                else:
                    gdata = gdata.flatten()
                my_cmap = cmap1 if gcolors is None else gcolors[group]
            else:
                my_cmap = None

            if group is None:
                posdf = pv.PolyData(adata.obsm[basis])
                gname=''
            else:
                posdf = pd.DataFrame(adata.obsm[basis], columns=['x','y','z'])
                posdf[group] = gdata
                gname = group

            if not save is None:
                save = save
            elif outpre:
                save = f'{outpre}.{gname}.{format}'
            else:
                save = outpre
            ipl = surfaces3d(meshlist = meshlist, 
                            points=posdf, 
                            groupby=group, 
                            title=f'{gname}',
                            point_cmap = my_cmap,
                            save = save,
                            **kargs)
            pls.append(ipl)
    return pls

class save_mesh:
    def __init__(self, save_type=None):
        self.save_type = save_type

    @staticmethod
    def save_movie(plotter, filename, framerate=24, view_up = None, viewup=(0, 0, 0),
                    factor=2, n_points=120, step=0.1, 
                    shift=0, focus=None, quality=10, write_frames=True):
        path = plotter.generate_orbital_path(factor=factor, shift=shift, viewup=view_up, n_points=n_points)
        if filename.endswith('gif'):
            plotter.open_gif(filename)
        else:
            plotter.open_movie(filename, framerate=framerate, quality=quality)
        plotter.orbit_on_path(path, write_frames=write_frames, viewup=viewup, step=step, focus=focus)

    @staticmethod
    def save_html(plotter, filename):
        plotter.export_html(filename)

    @staticmethod
    def save_vtksz(plotter, filename, format='zip'):
        plotter.export_vtksz(filename, format=format)

    def __call__(self, plotter, filename, show=False, format='zip', **kargs):
        if filename:
            save_type = filename.split('.')[-1] if self.save_type is None else self.save_type
            if save_type == 'mp4':
                self.save_movie(plotter, filename, **kargs)
            elif save_type == 'html':
                self.save_html(plotter, filename)
            elif save_type == 'vtksz':
                self.save_vtksz(plotter, filename)
            elif save_type == 'vrml':
                plotter.export_vrml(filename)
            elif save_type == 'obj':
                plotter.export_obj(filename)
            elif save_type == 'gltf':
                plotter.export_gltf(filename)
            elif save_type == 'vtk':
                plotter.save(filename)
        if show:
            plotter.show()
        elif show is False:
            plotter.close()
        return plotter

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)

def add_legend(
        self,
        labels=None,
        bcolor=(0.5, 0.5, 0.5),
        border=False,
        size=(0.2, 0.2),
        name=None,
        loc='upper right',
        face='triangle',
        font_family='courier',
    ):
    import pyvista as pv
    ###################### - Added - ########################
    legend_text = self._legend.GetEntryTextProperty()
    legend_text.SetFontFamily(pv.parse_font_family(font_family))
    #########################################################

    self.add_actor(self._legend, reset_camera=False, name=name, pickable=False)
    return self._legend
