import plotly.graph_objects as go
import numpy as np

class OptionPlotter:
    def __init__(self, greeks_df, strike_selection, premium_option, greek_selection, last_close):
        self.greeks_df = greeks_df
        self.strike_selection = strike_selection
        self.last_close = last_close
        self.premium_option = premium_option
        self.greek_selection = greek_selection

    def plot_long_call(self):
        fig = go.Figure()

        hover_text_pnl = [
            f"P&L: {pl:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            mode='lines',
            fill='tozeroy',
            fillgradient=dict(
                type='horizontal',
                colorscale=[[0, 'rgba(214, 56, 100, 0.5)'], [1, "rgba(86, 175, 156, 0.5)"]],
                start=self.strike_selection+self.premium_option-0.01,
                stop=self.strike_selection+self.premium_option,
            ),
            line=dict(color='rgba(15,17,22,0.3)'),
            name='P&L',
            hoverinfo='text',
            hovertext=hover_text_pnl
        ))

        hover_text_und = [
            f"Underlying Price: ${price:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            line=dict(color="rgba(15,17,22,0.3)"),
            hoverinfo='text',
            hovertext=hover_text_und
        ))

        fig.add_shape(
            type="line",
            x0=self.strike_selection,
            x1=self.strike_selection,
            y0=self.greeks_df['P&L'].min(),
            y1=self.greeks_df['P&L'].max(),
            line=dict(
                color="lightblue",
                width=2,
                dash="dash"
            ),
            name="Strike Price"
        )

        if self.greek_selection == 'Delta':
            hover_text_delta = [
                f"Delta: {delta:.2f}"
                for price, delta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Delta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Delta'],
                mode='lines',
                line=dict(color="orange", dash='dash'),
                name='Delta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_delta
            ))

        elif self.greek_selection == 'Gamma':
            hover_text_gamma = [
                f"Gamma: {gamma:.2f}"
                for price, gamma in zip(self.greeks_df['Underlying Price'], self.greeks_df['Gamma'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Gamma'],
                mode='lines',
                line=dict(color="pink", dash='dash'),
                name='Gamma',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_gamma
            ))
            
        elif self.greek_selection == 'Rho':
            hover_text_rho = [
                f"Rho: {rho:.2f}"
                for price, rho in zip(self.greeks_df['Underlying Price'], self.greeks_df['Rho'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Rho'],
                mode='lines',
                line=dict(color="red", dash='dash'),
                name='Rho',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_rho
            ))

        elif self.greek_selection == 'Theta':
            hover_text_theta = [
                f"Theta: {theta:.2f}"
                for price, theta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Theta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Theta'],
                mode='lines',
                line=dict(color="blue", dash='dash'),
                name='Theta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_theta
            ))

        elif self.greek_selection == 'Vega':
            hover_text_vega = [
                f"Vega: {vega:.2f}"
                for price, vega in zip(self.greeks_df['Underlying Price'], self.greeks_df['Vega'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Vega'],
                mode='lines',
                line=dict(color="yellow", dash='dash'),
                name='Vega',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_vega
            ))
        plot_data = self.greeks_df[:]

        if self.strike_selection < self.last_close:
            price_range = [np.floor((self.strike_selection/10)*0.8)*10, self.last_close*1.2]
        else:
            price_range = [np.floor((self.last_close/10)*0.8)*10, self.strike_selection*1.2]

        pl_range = [np.minimum(-1, -self.premium_option*1.1), max(price_range) - self.strike_selection - self.premium_option]

        fig.update_layout(
            xaxis=dict(title="Underlying Asset Price",
                #range=[plot_data['Underlying Price'].min(), plot_data['Underlying Price'].max()],
                range = price_range,
                showgrid=False,
                dtick=10),
            yaxis=dict(
                title="Profit & Loss",
                range=pl_range, #[np.minimum(-1, (-self.premium_option*1.1)), up_lim],
                showgrid=False),
            yaxis2= dict(
                title="Option " + self.greek_selection,
                overlaying='y',
                side='right',
                range=[plot_data[self.greek_selection].min(), plot_data[self.greek_selection].max()],
                showgrid=False),
            hovermode='x unified',
            showlegend=False,
            width=800,
            height=500,
            hoverlabel=dict(
                bgcolor="rgba(43,46,56,1)",
                font_size=14,
                font_color="white",
                font_family="Arial",
                bordercolor="white"),
            dragmode=False,
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            width=800,
            height=450
        )
        return fig

    def plot_long_put(self):
        fig = go.Figure()
        hover_text_pnl = [
            f"P&L: {pl:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            mode='lines',
            fill='tozeroy',
            fillgradient=dict(
                type='horizontal',
                colorscale=[[0, "rgba(86, 175, 156, 0.5)"], [1,  'rgba(214, 56, 100, 0.5)']],
                start=self.strike_selection-self.premium_option-0.01,
                stop=self.strike_selection-self.premium_option,
            ),
            line=dict(color='rgba(15,17,22,0.3)'),
            name='P&L',
            hoverinfo='text',
            hovertext=hover_text_pnl
        ))

        hover_text_und = [
            f"Underlying Price: ${price:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            line=dict(color="rgba(15,17,22,0.3)"),
            hoverinfo='text',
            hovertext=hover_text_und  # Custom hover text for P&L
        ))

        fig.add_shape(
            type="line",  # Type of shape is a line
            x0=self.strike_selection,  # Starting x position (strike price)
            x1=self.strike_selection,  # Ending x position (strike price, vertical line)
            y0=self.greeks_df['P&L'].min(),  # Start y position (minimum P&L value)
            y1=self.greeks_df['P&L'].max(),  # End y position (maximum P&L value)
            line=dict(
                color="white",  # Color of the line
                width=2,  # Width of the line
                dash="dash"  # Dash style of the line (dashed line)
            ),
            name="Strike Price"  # Optional: Name for the shape, can be used for reference
        )

        if self.greek_selection == 'Delta':
            hover_text_delta = [
                f"Delta: {delta:.2f}"
                for price, delta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Delta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Delta'],
                mode='lines',
                line=dict(color="orange", dash='dash'),
                name='Delta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_delta
            ))

        elif self.greek_selection == 'Gamma':
            hover_text_gamma = [
                f"Gamma: {gamma:.2f}"
                for price, gamma in zip(self.greeks_df['Underlying Price'], self.greeks_df['Gamma'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Gamma'],
                mode='lines',
                line=dict(color="pink", dash='dash'),
                name='Gamma',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_gamma
            ))
            
        elif self.greek_selection == 'Rho':
            hover_text_rho = [
                f"Rho: {rho:.2f}"
                for price, rho in zip(self.greeks_df['Underlying Price'], self.greeks_df['Rho'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Rho'],
                mode='lines',
                line=dict(color="red", dash='dash'),
                name='Rho',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_rho
            ))

        elif self.greek_selection == 'Theta':
            hover_text_theta = [
                f"Theta: {theta:.2f}"
                for price, theta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Theta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Theta'],
                mode='lines',
                line=dict(color="blue", dash='dash'),
                name='Theta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_theta
            ))

        elif self.greek_selection == 'Vega':
            hover_text_vega = [
                f"Vega: {vega:.2f}"
                for price, vega in zip(self.greeks_df['Underlying Price'], self.greeks_df['Vega'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Vega'],
                mode='lines',
                line=dict(color="yellow", dash='dash'),
                name='Vega',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_vega
            ))
        plot_data = self.greeks_df[:]

        if self.strike_selection < self.last_close:
            price_range = [np.floor((self.strike_selection/10)*0.8)*10, self.last_close*1.2]
        else:
            price_range = [np.floor((self.last_close/10)*0.8)*10, self.strike_selection*1.2]

        pl_range = [np.minimum(-1, -self.premium_option*1.1),self.strike_selection - min(price_range) - self.premium_option]

        fig.update_layout(
            xaxis=dict(title="Underlying Asset Price",
                range=price_range,#[plot_data['Underlying Price'].min(), plot_data['Underlying Price'].max()],
                showgrid=False,
                dtick=10),
            yaxis=dict(
                title="Profit & Loss",
                range=pl_range,#[-self.premium_option, plot_data['P&L'].max()],
                showgrid=False),
            yaxis2= dict(
                title="Option " + self.greek_selection,
                overlaying='y',
                side='right',
                range=[plot_data[self.greek_selection].min(), plot_data[self.greek_selection].max()],
                showgrid=False),
            hovermode='x unified',
            showlegend=False,
            width=800,
            height=500,
            hoverlabel=dict(
                bgcolor="rgba(43,46,56,1)",
                font_size=14,
                font_color="white",
                font_family="Arial",
                bordercolor="white"),
            dragmode=False,  # Disable dragging to zoom or pan
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            width=800,
            height=450
        )
        return fig
    
    def plot_short_put(self):
        fig = go.Figure()

        hover_text_pnl = [
            f"P&L: {pl:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            mode='lines',
            fill='tozeroy',
            fillgradient=dict(
                type='horizontal',
                colorscale=[[0, 'rgba(214, 56, 100, 0.5)'], [1, "rgba(86, 175, 156, 0.5)"]],
                start=self.strike_selection-self.premium_option-0.01,
                stop=self.strike_selection-self.premium_option,
            ),
            line=dict(color='rgba(15,17,22,0.3)'),
            name='P&L',
            hoverinfo='text',
            hovertext=hover_text_pnl
        ))

        hover_text_und = [
            f"Underlying Price: ${price:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            line=dict(color="rgba(15,17,22,0.3)"),
            hoverinfo='text',
            hovertext=hover_text_und
        ))

        fig.add_shape(
            type="line",
            x0=self.strike_selection,
            x1=self.strike_selection,
            y0=self.greeks_df['P&L'].min(),
            y1=self.greeks_df['P&L'].max(),
            line=dict(
                color="white",
                width=2,
                dash="dash"
            ),
            name="Strike Price"
        )

        if self.greek_selection == 'Delta':
            hover_text_delta = [
                f"Delta: {delta:.2f}"
                for price, delta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Delta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Delta'],
                mode='lines',
                line=dict(color="orange", dash='dash'),
                name='Delta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_delta
            ))

        elif self.greek_selection == 'Gamma':
            hover_text_gamma = [
                f"Gamma: {gamma:.2f}"
                for price, gamma in zip(self.greeks_df['Underlying Price'], self.greeks_df['Gamma'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Gamma'],
                mode='lines',
                line=dict(color="pink", dash='dash'),
                name='Gamma',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_gamma
            ))
            
        elif self.greek_selection == 'Rho':
            hover_text_rho = [
                f"Rho: {rho:.2f}"
                for price, rho in zip(self.greeks_df['Underlying Price'], self.greeks_df['Rho'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Rho'],
                mode='lines',
                line=dict(color="red", dash='dash'),
                name='Rho',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_rho
            ))

        elif self.greek_selection == 'Theta':
            hover_text_theta = [
                f"Theta: {theta:.2f}"
                for price, theta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Theta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Theta'],
                mode='lines',
                line=dict(color="blue", dash='dash'),
                name='Theta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_theta
            ))

        elif self.greek_selection == 'Vega':
            hover_text_vega = [
                f"Vega: {vega:.2f}"
                for price, vega in zip(self.greeks_df['Underlying Price'], self.greeks_df['Vega'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Vega'],
                mode='lines',
                line=dict(color="yellow", dash='dash'),
                name='Vega',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_vega
            ))

        plot_data = self.greeks_df[:]

        if self.strike_selection < self.last_close:
            price_range = [np.floor((self.strike_selection/10)*0.8)*10, self.last_close*1.2]
        else:
            price_range = [np.floor((self.last_close/10)*0.8)*10, self.strike_selection*1.2]
            #self.greeks_df['P&L'] = self.premium_option - np.maximum(self.strike_selection - self.greeks_df['Underlying Price'], 0)

        pl_range = [self.premium_option - np.maximum(self.strike_selection-min(price_range), 0), self.premium_option]


        fig.update_layout(
            xaxis=dict(title="Underlying Asset Price",
                #range=[plot_data['Underlying Price'].min(), plot_data['Underlying Price'].max()],
                range=price_range,
                showgrid=False,
                dtick=10),
            yaxis=dict(
                title="Profit & Loss",
                range=pl_range,
                #range=[-self.premium_option, plot_data['P&L'].max()*1.1],
                showgrid=False),
            yaxis2= dict(
                title="Option " + self.greek_selection,
                overlaying='y',
                side='right',
                range=[plot_data[self.greek_selection].min(), plot_data[self.greek_selection].max()],
                showgrid=False),
            hovermode='x unified',
            showlegend=False,
            width=800,
            height=500,
            hoverlabel=dict(
                bgcolor="rgba(43,46,56,1)",
                font_size=14,
                font_color="white",
                font_family="Arial",
                bordercolor="white"),
            dragmode=False,
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            width=800,
            height=450
        )
        return fig
    
    def plot_short_call(self):
        fig = go.Figure()
        hover_text_pnl = [
            f"P&L: {pl:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            mode='lines',
            fill='tozeroy',
            fillgradient=dict(
                type='horizontal',
                colorscale=[[0, "rgba(86, 175, 156, 0.5)"], [1,  'rgba(214, 56, 100, 0.5)']],
                start=self.strike_selection+self.premium_option-0.01,
                stop=self.strike_selection+self.premium_option,
            ),
            line=dict(color='rgba(15,17,22,0.3)'),
            name='P&L',
            hoverinfo='text',
            hovertext=hover_text_pnl
        ))

        hover_text_und = [
            f"Underlying Price: ${price:.2f}"
            for price, pl in zip(self.greeks_df['Underlying Price'], self.greeks_df['P&L'])
        ]
        fig.add_trace(go.Scatter(
            x=self.greeks_df['Underlying Price'],
            y=self.greeks_df['P&L'],
            line=dict(color="rgba(15,17,22,0.3)"),
            hoverinfo='text',
            hovertext=hover_text_und  # Custom hover text for P&L
        ))

        fig.add_shape(
            type="line",  # Type of shape is a line
            x0=self.strike_selection,  # Starting x position (strike price)
            x1=self.strike_selection,  # Ending x position (strike price, vertical line)
            y0=self.greeks_df['P&L'].min(),  # Start y position (minimum P&L value)
            y1=self.greeks_df['P&L'].max(),  # End y position (maximum P&L value)
            line=dict(
                color="white",  # Color of the line
                width=2,  # Width of the line
                dash="dash"  # Dash style of the line (dashed line)
            ),
            name="Strike Price"  # Optional: Name for the shape, can be used for reference
        )

        if self.greek_selection == 'Delta':
            hover_text_delta = [
                f"Delta: {delta:.2f}"
                for price, delta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Delta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Delta'],
                mode='lines',
                line=dict(color="orange", dash='dash'),
                name='Delta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_delta
            ))

        elif self.greek_selection == 'Gamma':
            hover_text_gamma = [
                f"Gamma: {gamma:.2f}"
                for price, gamma in zip(self.greeks_df['Underlying Price'], self.greeks_df['Gamma'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Gamma'],
                mode='lines',
                line=dict(color="pink", dash='dash'),
                name='Gamma',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_gamma
            ))
            
        elif self.greek_selection == 'Rho':
            hover_text_rho = [
                f"Rho: {rho:.2f}"
                for price, rho in zip(self.greeks_df['Underlying Price'], self.greeks_df['Rho'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Rho'],
                mode='lines',
                line=dict(color="red", dash='dash'),
                name='Rho',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_rho
            ))

        elif self.greek_selection == 'Theta':
            hover_text_theta = [
                f"Theta: {theta:.2f}"
                for price, theta in zip(self.greeks_df['Underlying Price'], self.greeks_df['Theta'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Theta'],
                mode='lines',
                line=dict(color="blue", dash='dash'),
                name='Theta',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_theta
            ))

        elif self.greek_selection == 'Vega':
            hover_text_vega = [
                f"Vega: {vega:.2f}"
                for price, vega in zip(self.greeks_df['Underlying Price'], self.greeks_df['Vega'])
            ]

            fig.add_trace(go.Scatter(
                x=self.greeks_df['Underlying Price'],
                y=self.greeks_df['Vega'],
                mode='lines',
                line=dict(color="yellow", dash='dash'),
                name='Vega',
                yaxis='y2',
                hoverinfo='text',
                hovertext=hover_text_vega
            ))
        plot_data = self.greeks_df[:]
        #self.greeks_df['P&L'] = self.premium_option - np.maximum(self.greeks_df['Underlying Price'] - self.strike_selection, 0)

        if self.strike_selection < self.last_close:
            price_range = [np.floor((self.strike_selection/10)*0.8)*10, self.last_close*1.2]
        else:
            price_range = [np.floor((self.last_close/10)*0.8)*10, self.strike_selection*1.2]

        pl_range = [self.premium_option - np.maximum(max(price_range)-self.strike_selection, 0), self.premium_option]
        fig.update_layout(
            xaxis=dict(title="Underlying Asset Price",
                range=price_range,#[plot_data['Underlying Price'].min(), plot_data['Underlying Price'].max()],
                showgrid=False,
                dtick=10),
            yaxis=dict(
                title="Profit & Loss",
                range=pl_range,#[-self.premium_option, plot_data['P&L'].max()],
                showgrid=False),
            yaxis2= dict(
                title="Option " + self.greek_selection,
                overlaying='y',
                side='right',
                range=[plot_data[self.greek_selection].min(), plot_data[self.greek_selection].max()],
                showgrid=False),
            hovermode='x unified',
            showlegend=False,
            width=800,
            height=500,
            hoverlabel=dict(
                bgcolor="rgba(43,46,56,1)",
                font_size=14,
                font_color="white",
                font_family="Arial",
                bordercolor="white"),
            dragmode=False,  # Disable dragging to zoom or pan
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            autosize=True,
            width=800,
            height=450
        )
        return fig