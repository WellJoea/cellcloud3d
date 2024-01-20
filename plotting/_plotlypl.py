import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scanpy as sc
import plotly

from cellcloud3d.plotting._utilis import colrows, pdtype
from cellcloud3d.plotting._colors import color_palette, adata_color, pxsetcolor,vartype,cmap1px, cmaptolist
from cellcloud3d.plotting._spatial3d import get_spatial_info
from cellcloud3d.plotting._utilis import image2batyes

class plty():
    def __init__(self, adata = None,
                    groups=None, group_df=None,
                    corr_df = None, use_raw=False,
                    basis='X_umap', dimsname=None,
                    cmap = None, palette = None,
                    ignore_error=False,
                    image = None,
                    img_key=None, 
                    library_id=None, 
                    get_pix_loc=False, 
                    rescale=None,
                    update_loc=True,
                    sample = None,
                    seed = 200504,
                    axes = None):
        np.random.seed(491001)
        if sample:
            obs_idex = np.random.choice(range(adata.shape[0]), 
                            size=int(adata.shape[0]*sample), 
                            replace=False, p=None)
            self.adata = adata[obs_idex].copy()
        else:
            self.adata = adata.copy()
        self.fetchdata(adata=self.adata.copy(), 
                        groups=groups, group_df=group_df,
                        corr_df = corr_df, use_raw=use_raw,
                        basis= basis, dimsname=dimsname,
                        cmap = cmap, palette = palette,
                        axes = axes)
        self.fetchimage(adata=self.adata.copy(), image = image,
                        img_key=img_key, basis = basis, 
                        library_id=library_id, 
                        get_pix_loc=get_pix_loc,
                        update_loc=update_loc,
                        rescale=rescale)

    def fetchdata(self, adata=None, groups=None, group_df=None,
                    corr_df = None, use_raw=False,
                    basis='X_umap', dimsname=None,
                    cmap = None, palette = None,
                    axes = None,
                     **kargs):
        if (not adata is None) and use_raw:
            adata = adata.raw.to_adata().copy()

        if corr_df is None:
            try:
                corr_df = pd.DataFrame(adata.obsm[basis],
                                        index =adata.obs_names).copy()
                corr_df.columns = [ f"{basis.strip('X_')}_{i}" for i in range(corr_df.shape[1])]
            except ValueError:
                print("Error: Please input valid basis name or corr_df.")
        else:
            corr_df = corr_df.copy()

        if axes is not None:
            self.dims = len(axes)
            self.axes = axes
        else:
            self.dims = corr_df.shape[1]
            self.axes  = range(self.dims)

        if group_df is None:
            groups = [groups] if type(groups) in [str, int, float, bool] else groups
            groups = list(dict.fromkeys(groups))
            var_groups = adata.var_names.intersection(groups).tolist()
            obs_groups = adata.obs.columns.intersection(groups).tolist()

            group_df = []
            if len(obs_groups)>0:
                group_df.append(adata.obs[obs_groups])
            if len(var_groups)>0:
                try:
                    var_arr = adata[:,var_groups].X.toarray()
                except:
                    var_arr = adata[:,var_groups].X
                var_arr = pd.DataFrame(var_arr, 
                                        index=adata.obs_names,
                                        columns=var_groups)
                group_df.append(var_arr)
            assert len(group_df)>0, 'No group was fond in adata.'
            group_df = pd.concat(group_df, axis=1)
            groups  = [i for i in groups if i in group_df.columns.tolist()]
            group_df = group_df[groups]

        common = np.intersect1d(corr_df.index, group_df.index)
        corr_df = corr_df.loc[common,:].iloc[:, self.axes]
        group_df= group_df.loc[common,:]

        dimsname = corr_df.columns.tolist() if dimsname is None else dimsname
        corr_df.columns = dimsname

        colors = {}
        gdtype = {}
        #print(group_df.head(), group_df.dtypes)
        for col in group_df.columns:
            igroup = group_df[col]
            gtype = pdtype(igroup)
            if gtype == 'discrete':
                try:
                    iorders = igroup.cat.categories.tolist()
                except:
                    iorders = igroup.unique().tolist()
                group_df[col] = pd.Categorical(igroup, categories=iorders)
    
                if type(palette) in [list, np.ndarray]:
                    icolor = palette
                elif type(cmap) in [dict]:
                    icolor = palette[col]
                else:
                    if adata is None:
                        icolor = color_palette(len(iorders))
                    else:
                        iuns_cor = f'{col}_colors'
                        if (iuns_cor in adata.uns.keys()) and len(iorders)<= len(adata.uns[iuns_cor]):
                            icolor = adata.uns[iuns_cor]
                        else:
                            adata.obs[col] = pd.Categorical(igroup, categories=iorders)
                            adata_color(adata, value = col, cmap=cmap, palette=palette) 
                            icolor = adata.uns[iuns_cor]
            else:
                if type(cmap) == str:
                    icolor = cmap
                # elif type(cmap) in [list, np.ndarray]:
                #     cmap = list(cmap)
                #     icolor = cmap.pop(0)
                elif type(cmap) in [dict]:
                    icolor = cmap[col]
                elif cmap is None:
                    icolor = cmap1px
                else:
                    icolor = cmap
            colors[col] = icolor
            gdtype[col] = gtype
        self.dimsname = dimsname
        self.dims = corr_df.shape[1]
        self.colors = colors
        self.gdtype = gdtype
        self.corr_df = corr_df
        self.group_df = group_df
        self.groups = group_df.columns.tolist()
        self.data = pd.concat([corr_df, group_df], axis=1)
        self.data['index'] = self.data.index
        self.ngroup = group_df.shape[1]
        return self

    def fetchimage(self, adata=None, image = None,
                    img_key=None, basis = None, 
                    update_loc=True,
                    library_id=None, get_pix_loc=False, rescale=None):
        if img_key:
            sinfo = get_spatial_info(adata, 
                                    img_key=img_key, 
                                    basis = basis, 
                                    library_id=library_id, 
                                    get_pix_loc=get_pix_loc, 
                                    rescale=rescale)
            image = sinfo.get('img')
            st_loc = sinfo.get('locs')
            rescale = sinfo.get('rescale',1)
            if update_loc:
                st_loc = st_loc.loc[self.corr_df.index, :]
                st_loc.columns = self.corr_df.columns
                self.corr_df = st_loc
                self.data = pd.concat([self.corr_df, self.group_df], axis=1)

        self.image = image
        return self

    def scatter2ds(self, data=None, save=None, outpre=None, size=None, size_max =20, scale=1, show=False,
                   template='none', scene_aspectmode='data',
                   y_reverse=False, x_reverse=False,  same_scale=True, 
                   scene_aspectratio=dict(x=1, y=1), return_fig =False,
                   ncols=2, soft=False, figscale=None, error=20, 
                   clips = None, showlegend=True,
                   show_grid =True, 
                   showticklabels = False,
                   itemsizing='constant',
                   show_image=True,
                   imageaspos=False,
                   sample=None,
                   image_opacity = None,
                   cmap=None,
                   random_state=19491001,
                   legdict={}, **kargs, ):
        nrows, ncols = colrows(self.ngroup, ncols=ncols, soft=soft)
        dimdict= dict(zip(list('xy'), self.dimsname))
        data_df = self.data.copy() if data is None else data.copy()
        if not sample is None:
            if (type(sample) == int) and (sample>1):
                data_df = data_df.sample(n=sample, 
                                        replace=False,
                                        random_state=random_state)
            if (type(sample) == float) and (sample<=1):
                data_df = data_df.sample(frac=sample, 
                                        replace=False,
                                        random_state=random_state)

        clips = None if clips is None else list(clips)
        width = None if figscale is None else ncols*figscale+error 
        height= None if figscale is None else nrows*figscale 
        cmap = cmap1px if cmap is None else cmap

        category_orders = []
        color_discrete_sequence = []
        dis_groups = []
        con_groups = []
        for col in self.groups:
            gtype = self.gdtype[col]
            if gtype=='discrete':
                category_orders.extend(data_df[col].cat.categories.tolist())
                color_discrete_sequence.extend(self.colors[col])
                dis_groups.append(col)
            elif gtype =='continuous':
                if not clips is None:
                    data_df[col] = np.clip(data_df[col], clips[0], clips[1])
                if np.ndim(clips)==1:
                    data_df[col] = np.clip(data_df[col], clips[0], clips[1])
                elif np.ndim(clips) > 1:
                    data_df[col] = np.clip(data_df[col], clips.pop(0)[0], clips.pop(0)[1])
                con_groups.append(col)
        # if len(dis_groups)<1:
        #     raise ValueError('Only discrete groups will be ploted!!')

        #plot
        SData = pd.melt(data_df, id_vars= self.dimsname, 
                                value_vars=dis_groups+ con_groups,
                                var_name='groups', value_name='Type')
        SData['index'] = SData.index
        fig = px.scatter(SData, color="Type", 
                         facet_col="groups", 
                         facet_col_wrap=ncols,
                         size = size,
                         size_max=size_max,
                         width=width, height=height,
                         color_discrete_sequence=color_discrete_sequence,
                         color_continuous_scale = cmap,
                         hover_name="index", hover_data=["index"],
                         category_orders={'Type': category_orders},
                         **dimdict, **kargs)
        # for i in range(len(fig.data)):
        #     if hasattr(fig.data[i], 'showlegend'):
        #         fig.data[i].showlegend = True

        fig.update_layout(legend=dict( itemsizing = itemsizing, itemwidth=30+len(dis_groups), **legdict),
                          showlegend=showlegend,
                          scene_aspectmode=scene_aspectmode,
                          template=template,
                          scene_aspectratio=scene_aspectratio,
                            scene=dict(
                                xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_bgcolor='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        if not size is None:
            fig.update_traces(marker=dict(line=dict(width=0,color='DarkSlateGrey')),
                              selector=dict(mode='markers'))
        else:
            fig.update_traces(marker=dict(size=scale, line=dict(width=0,color='DarkSlateGrey')),
                              selector=dict(mode='markers'))
        if same_scale:
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )

        if show_image and (not self.image is None):
            if imageaspos:
                self.add_image(fig, self.image, image_opacity=image_opacity,)
            else:
                self.add_layout_image(fig, self.image,
                            image_opacity=image_opacity,
                            x_reverse=x_reverse,
                            y_reverse=y_reverse)

        if y_reverse:
            fig.update_yaxes(autorange="reversed")
        if x_reverse:
            fig.update_xaxes(autorange="reversed")

        if show:
            fig.show()
        if outpre or save:
            save= outpre + '.'+ '.'.join(dis_groups) + '.2d.html' if outpre else save
            fig.write_html(save)
        if return_fig:
            return fig

    def scatter3d(self, group, data=None, select=None, nacolor='lightgrey', show=False,
                  xyz= [0,1,2],
                  size=None, size_max=20, scale=1, width=None, height=None,
                  scene_aspectmode='data', keep_all=False, order=True, 
                  ascending=False, return_fig =False, 
                  show_grid =True, 
                  showticklabels = False,
                  y_reverse=False, x_reverse=False, z_reverse=False,
                  scene_aspectratio=dict(x=1, y=1, z=1),
                  itemsizing='constant',
                  template='none', save=None, 
                  clip = None,
                  sample=None,
                  random_state=19491001,
                  legdict={}, laydict={}, **kargs):
        if self.dims <3:
            raise ValueError('The dims must be larger than 2!!')
        dimdict = dict(zip(list('xyz'), np.array(self.dimsname)[xyz]))
        # ctype
        ctype = self.gdtype[group]
        color = self.colors[group]
        idata = self.data.copy() if data is None else data.copy()
        idata['index'] = idata.index
        if not sample is None:
            if (type(sample) == int) and (sample>1):
                idata = idata.sample(n=sample, 
                                        replace=False,
                                        random_state=random_state)
            if (type(sample) == float) and (sample<=1):
                idata = idata.sample(frac=sample, 
                                        replace=False,
                                        random_state=random_state)

        dimdict.update(pxsetcolor(color, ctype=ctype))
        # order
        if ctype == 'discrete':
            order = idata[group].cat.categories.tolist()
            if not keep_all:
                idata[group] = idata[group].cat.remove_unused_categories()
                keep_order = idata[group].cat.categories.tolist()
                colors = dimdict['color_discrete_sequence']
                if type(colors)==list:
                    colors =[c for c,o in zip(colors, order) if o in keep_order]
                    dimdict['color_discrete_sequence'] = colors
                order = keep_order
            category_orders={group: order}
            if not select is None:
                select = [select] if type(select) == str else select
                if type(dimdict['color_discrete_sequence'])==list:
                    colors = [  cs if co in select else nacolor
                                for co,cs in  zip(category_orders[group], dimdict['color_discrete_sequence']) ]
                    dimdict['color_discrete_sequence'] = colors
            dimdict.update({'category_orders': category_orders}) #'animation_frame': group

        elif ctype == 'continuous':
            if not clip is None:
                idata[group] = np.clip(idata[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                idata.sort_values(by = group, ascending=ascending, inplace=True)
        fig = px.scatter_3d(idata, 
                            color=group, 
                            size=size,
                            size_max=size_max,
                            width=width, height=height,
                            hover_name="index", hover_data=["index"],
                             **dimdict, **kargs)
        
        fig.update_layout(legend=dict(itemsizing = itemsizing, **legdict),
                          scene_aspectmode=scene_aspectmode,
                          scene_aspectratio=scene_aspectratio,
                          template=template,
                            scene=dict(
                                xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            ),
                          plot_bgcolor='#FFFFFF',) #
                          #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_or='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)
        if not size is None:
            fig.update_traces(marker=dict(line=dict(width=0,color='DarkSlateGrey')),
                              selector=dict(mode='markers'))
        else:
            fig.update_traces(marker=dict(size=scale, line=dict(width=0,color='DarkSlateGrey')),
                              selector=dict(mode='markers'))
        if y_reverse:
            fig.update_yaxes(autorange="reversed")
        if x_reverse:
            fig.update_xaxes(autorange="reversed")
        if z_reverse:
            fig.update_zaxes(autorange="reversed")
        if show:
            fig.show()
        if save:
            fig.write_html(save)
        if return_fig:
            return fig

    def scatter3ds(self, groups=None, outpre=None, clips = None,  **kargs):
        groups = self.groups if (groups is None) else groups
        clips = None if clips is None else list(clips)
        for i in groups:
            if (clips is None) or (self.gdtype[i] != 'continuous'):
                clip = None
            else:
                if np.ndim(clips)==1:
                    clip = clips
                else:
                    clip = clips.pop(0)
            save = None if outpre is None else '%s.%s.3d.html'%(outpre, i.replace(' ','.'))
            self.scatter3d(i, save=save, clip = clip, **kargs)

    def scatters(self, groups=None, data = None,  matches = None, 
                    line_color='gray',  line_width=1, line_alpha = 1, color='b',
                   line_weight = None, line_cmap = None,
                 out=None, outpre=None, show=False, ncols=2, figscale=800, werror=30, 
                   aspectmode='data', shared_xaxes=True, shared_yaxes=True, 
                   aspectratio=dict(x=1, y=1, z=1), y_reverse=False, x_reverse=False,
                   xyz = [0,1,2],
                   clips = None, image_opacity=None,
                   error=10, scale=1, legendwscape=0.1, lengedxloc = 1.05, keep_all=False,
                   ascending=False, return_fig =False, legend_tracegroupgap = 25, legend_font_size=14,
                   thickness=20, cblen=0.55, cby=0.5, ticks='outside', tickmode='auto', template='none',
                   order ='guess', soft=False, 
                   clickmode='event',
                   same_scale=True, 
                   show_grid =True, 
                   showticklabels = False,
                   legend_groupclick='toggleitem',
                   legend_itemclick='toggle',
                   legend_itemdoubleclick='toggleothers',
                   show_image=False,
                   imageaspos=False, 
                   sample=None,
                   random_state=19491001,
                   margin=None,
                   showlegend=True,
                   showscale =True,
                   cmap=None, 
                   subplot_dict = {}, 
                   **kargs):
        groups= self.groups if (groups is None) else groups
        clips = None if clips is None else list(clips)
        idata = self.data.copy() if data is None else data.copy()
        idata['index'] = idata.index
        if not sample is None:
            if (type(sample) == int) and (sample>1):
                idata = idata.sample(n=sample, 
                                        replace=False,
                                        random_state=random_state)
            if (type(sample) == float) and (sample<=1):
                idata = idata.sample(frac=sample, 
                                        replace=False,
                                        random_state=random_state)

        ncell = len(groups)
        nrows, ncols = colrows(ncell, ncols=ncols, soft=soft)
        height=None if figscale is None else figscale*nrows
        width= None if figscale is None else (werror+figscale)*ncols

        if self.dims==2:
            GoS = go.Scatter
            fig = make_subplots(rows=nrows, cols=ncols, 
                                shared_xaxes=shared_xaxes, 
                                shared_yaxes=shared_yaxes, 
                                print_grid =False,
                                subplot_titles=groups,
                                **subplot_dict)
        elif self.dims==3:
            GoS = go.Scatter3d
            specs = np.array([{"type": "scene"},]*(nrows * ncols)).reshape(nrows, ncols)
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=groups, specs=specs.tolist())

        legendps = lengedxloc+legendwscape if 'discrete' in list(self.gdtype.values()) else lengedxloc
        for n,group in enumerate(groups):
            irow, icol = n//ncols+1, n%ncols+1
            ctype = self.gdtype[group]
            color = self.colors[group]
            colors = pxsetcolor(color, ctype=ctype)

            # plot
            if ctype == 'discrete':
                order = idata[group].cat.categories.tolist()
                if not keep_all:
                    idata[group] = idata[group].cat.remove_unused_categories()
                    keep_order = idata[group].cat.categories.tolist()
                    if type(color)==list:
                        color =[c for c,o in zip(color, order) if o in keep_order]
                    order = keep_order
                cordict = dict(zip(order,color))
                for _n in order:
                    iidata  = idata[idata[group]==_n]
                    dimdict = { i[0]: iidata[i[1]] for i in zip(list('xyz'), self.dimsname) }
                    dimdict.update({'name':_n, 'legendgrouptitle':{'text':group, 'font': {'size':14}}, 'legendgroup' : str(n+1)})
                    #if Colors:
                    dimdict.update({'marker': dict(color=cordict[_n],
                                                   size=scale, 
                                                   line=dict(width=0,color='DarkSlateGrey'))})
                    fig.append_trace(GoS(mode="markers", showlegend=showlegend,  **dimdict, **kargs), row=irow, col=icol)
            elif ctype == 'continuous':
                if np.ndim(clips)==1:
                    idata[group] = np.clip(idata[group], clips[0], clips[1])
                elif np.ndim(clips) > 1:
                    idata[group] = np.clip(idata[group], clips.pop(0)[0], clips.pop(0)[1])
                if order =='guess' or  order == True:
                    idata.sort_values(by = group, ascending=ascending, inplace=True)
                dimdict = { i[0]: idata[i[1]] for i in zip(list('xyz'), self.dimsname)}
                dimdict.update({'name': group, 'legendgroup' : str(n+1)})
                colorscale = colors['color_continuous_scale'] if cmap is None else cmap
                colorbar=dict(thickness=thickness, title=group,
                            len=cblen, x=legendps,y=cby, 
                            tickmode=tickmode,
                            ticks= ticks,
                            outlinewidth=0) if showlegend else None

                dimdict.update({'marker': dict(colorscale=colorscale, 
                                               showscale=showscale,
                                               color=idata[group],
                                               size=scale,
                                               line=dict(width=0,color='DarkSlateGrey'),
                                               colorbar=colorbar)})
                fig.append_trace(GoS(mode="markers", showlegend=showlegend, marker_coloraxis=None, 
                                     hover_name="index", hover_data=["index"],
                                     **dimdict, **kargs), row=irow, col=icol)
                legendps += legendwscape

            if same_scale:
                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                    row=irow, col=icol,
                )
            fig.update_scenes(aspectmode=aspectmode, 
                              aspectratio=aspectratio,
                              row=irow, col=icol)
        fig.update_traces(marker=dict(size=scale, line=dict(width=0,color='DarkSlateGrey')),
                          showlegend=showlegend,
                          selector=dict(mode='markers'))

        if not matches is None:
            assert matches.shape[1] >= 6
            if not line_weight is None:
                line_weight = np.array(line_weight)
                line_weight = line_weight/line_weight.max()
                line_widths = line_weight * line_width
            else:
                line_widths = np.ones(matches.shape[0]) * line_width

            if not line_weight is None:
                line_colors = cmaptolist(line_cmap, spaces=line_weight)
            else:
                line_colors = [line_color] * matches.shape[0]

            for i in range(matches.shape[0]):
                fig.append_trace(GoS( x=matches[i,[0,3]], y=matches[i,[1,4]], z=matches[i,[2,5]], 
                                    mode="lines", showlegend = False,
                                    line={'color': line_color, 
                                          'width': line_width},),
                                    row=irow, col=icol)

        fig.update_layout(
                          height=height, width=width,
                          #showlegend=showlegend,
                          #scene_aspectmode=aspectmode,
                          #scene_aspectratio=aspectratio,
                          legend=dict( itemsizing = 'constant', itemwidth=30+len(groups)),
                          legend_tracegroupgap = legend_tracegroupgap,
                          legend_groupclick=legend_groupclick,
                          legend_itemclick=legend_itemclick,
                          legend_itemdoubleclick=legend_itemdoubleclick,
                          legend_font_size=legend_font_size,
                          clickmode=clickmode,
                          template=template,
                            scene=dict(
                                xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                                aspectmode=aspectmode, 
                              aspectratio=aspectratio,
                            ),
                          #autosize=False,
                          margin=margin,
                          plot_bgcolor='#FFFFFF',) 
                          #template='simple_white', 
                          #paper_bgcolor='#000000',
                          #plot_or='#000000'
                          #fig.update_xaxes(visible=False, showticklabels=False)

        if show_image and (not self.image is None):
            if imageaspos:
                self.add_image(fig, self.image, image_opacity=image_opacity,)
            else:
                self.add_layout_image(fig, self.image,
                            image_opacity=image_opacity,
                            x_reverse=x_reverse,
                            y_reverse=y_reverse)

        #fig = go.FigureWidget(fig)
        if y_reverse:
            fig.update_yaxes(autorange="reversed")
        if x_reverse:
            fig.update_xaxes(autorange="reversed")

        if show:
            fig.show()
        if outpre or out :
            out = '%s.%s.%sd.html'%(outpre, '.'.join(groups), self.dims) if outpre else out
            fig.write_html(out)
        if return_fig:
            return fig

    @staticmethod
    def add_layout_image(fig, image, x_reverse=False, y_reverse=False, 
                         sizing='stretch', image_opacity=None,
                         **kargs):
        imagebt = image2batyes(image)
        sizey, sizex = image.shape[:2]
        x = sizex if x_reverse else 0
        y = sizey if y_reverse else 0

        xyaxis = []
        for i in fig.data:
            ix, iy =  i['xaxis'], i['yaxis']
            if not [ix, iy] in xyaxis:
                xyaxis.append([ix, iy])
                fig.add_layout_image(
                    source=imagebt,
                    xref=ix,
                    yref=iy,
                    x=x, 
                    y=y,
                    xanchor="left",
                    yanchor="bottom",
                    layer="below",
                    sizing=sizing,
                    sizex=sizex,
                    sizey=sizey,
                    opacity = image_opacity,
                    **kargs
                )

    @staticmethod
    def add_image(fig, image, image_opacity=None,**kargs):
        imagebt = image2batyes(image)
        xyaxis = []
        for i in fig.data:
            ix, iy =  i['xaxis'], i['yaxis']
            if not [ix, iy] in xyaxis:
                xyaxis.append([ix, iy])
                fig.add_image(
                    source=imagebt,
                    xaxis =ix,
                    yaxis =iy,
                    opacity=image_opacity,
                    **kargs
                )

        # for i in fig.data:
        #     print(ixy)
        #     if type(i) == plotly.graph_objs._scattergl.Scattergl:
        #         fig.add_image(
        #             source=imagebt,
        #             xaxis =f'x{ixy}',
        #             yaxis =f'y{ixy}', 
        #             opacity=image_opacity,
        #             **kargs
        #         )
        #         ixy +=1

def qscatter(Data, X=0, Y=1, Z=2, group=3, save=None, show=True, scale=1,
                xwidth=800, ywidth=800, zwidth=800, scene_aspectmode='data',
                scene_aspectratio=dict(x=1, y=1, z=1), clip=None, sample=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                colormap=None, template='none', **kargs):
    if isinstance(Data, np.ndarray):
        Data = pd.DataFrame(Data, columns=[0,1,2])
    Data['index'] = Data.index

    if isinstance(group, pd.Series) or \
        isinstance(group, pd.core.arrays.categorical.Categorical) or \
        isinstance(group, np.ndarray) or \
        isinstance(group, list):
        try:
            Order = group.cat.categories
            Data['group'] = pd.Categorical(np.array(group), categories=Order)
        except:
            Data['group'] = np.array(group)
        group = 'group'
    if not sample is None:
        if (type(sample) == int) and (sample>1):
            Data = Data.sample(n=sample, 
                                    replace=False,
                                    random_state=random_state)
        if (type(sample) == float) and (sample<=1):
            Data = Data.sample(frac=sample, 
                                    replace=False,
                                    random_state=random_state)

    dimdict = dict(zip(list('xyz'), (X,Y,Z)))
    ctype   = vartype(Data[group])

    if ctype == 'discrete':
        try:
            order = Data[group].cat.categories.tolist()
        except:
            order = Data[group].unique().tolist()
        color_seq = 'color_discrete_sequence'
        colormap = color_palette(len(order))
        category_orders={group: order, color_seq : colormap}
        dimdict.update({'category_orders': category_orders}) #'animation_frame': group
    elif ctype == 'continuous':
        Data[group] = np.clip(Data[group], clip[0], clip[1])
        if order =='guess' or  order == True:
            Data = Data.sort_values(by = group, ascending=True)
        color_seq = 'color_continuous_scale'
        colormap = 'Viridis' if colormap is None else colormap
        dimdict.update({color_seq:colormap})

    fig = px.scatter_3d(Data, color=group, hover_name="index", hover_data=["index"],
                         **dimdict, **kargs) #width=width, height=height,
    fig.update_layout(legend=dict(itemsizing = 'constant'),
                    scene_aspectmode=scene_aspectmode,
                        scene_aspectratio=scene_aspectratio,
                        template=template,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        plot_bgcolor='#FFFFFF',) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_traces(marker=dict(size=scale,
                        line=dict(width=0,color='DarkSlateGrey')),
                        selector=dict(mode='markers'))

    fig.update_traces(#hovertemplate="Sepal Width: %{x}<br>Sepal Length: %{y}<br>%{text}<br>Petal Width: %{customdata[1]}",
                    text=[{"index": Data.index}])

    if show:
        fig.show()
    if save:
        fig.write_html(save)