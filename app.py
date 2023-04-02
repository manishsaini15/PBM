import io
import zipfile
from flask import Flask, render_template, request, url_for, redirect,make_response
import numpy as np
import pandas as pd
import math
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.offline import plot


app = Flask(__name__)
def run_simulation(t_start, t_end, v_min, v_max, r,arr,initial_condition1,  aggregation_kernel, daughter_distribution, breakage_kernel):
    #  setting the paramter
    t_start=float(t_start)
    t_end=float(t_end)
    v_min=float(v_min)
    v_max=float(v_max)
    r=float(r)
    N=  abs(math.log10(v_max/v_min) / math.log10(r))  + 1
    m_v= math.ceil(N)  # number of volume nodes
    m= m_v -1  # number of pivots
    tspan = np.array([t_start, t_end])

    # ---------setting the meshes in bin width array, pivots array, bin boundary ------

    vi=np.zeros(m_v)    #bin boundary array
    x=np.zeros(m)       # pivots array
    width=np.zeros(m)   # width array
    vi[0]=v_min
    vi[m_v-1]=v_max
    for i in range(1,m_v-1):
        # vi[m_v-1-i]=  v_min * pow(r,N-i-1) 
        # vi[i]= v_min + pow(2, (i- m_v +1)/q) *(v_max -v_min)
        vi[i]=vi[i-1]*r
    for i in range(m):
        x[i]= (vi[i+1] + vi[i])/ 2 
        width[i]=  vi[i+1]-vi[i]  

    # ------------setting up the initial condtions // initial  number of particle on the pivots
    Nstart=np.zeros(m)
    try :
        if float(initial_condition1):
            Nstart[m-1]= float(initial_condition1)
    except ValueError:
        def initial(y):
            return eval(initial_condition1)
        for i in range(m):
            Nstart[i]= quad(initial,vi[i],vi[i+1])[0]
            # Nstart[i]= - math.exp(-vi[i+1]) + math.exp(-vi[i])
    #   ---------------------kernel function  -----------------------------------------
    def breakage_rate(y):
        return eval(breakage_kernel)

    def daughter_dist (y):
        return eval(daughter_distribution)
    def aggr_kernel(x=0,y=0):
        return eval(aggregation_kernel)
    # ------------- Breakage rate matrics ----------------------------------------------
    Gam=np.zeros(m)
    for i in range (m):
        Gam[i]=breakage_rate(x[i])
    
    # Genrating n(i,k) matrix it is a matrix that provides the breakge birth to population at ith #pivot due to the breakage of the on the kth pivot
    
    n = np.zeros((m,m))   # n(i,k)  matrics for breakage
    # The second integral is zero for the first pivot
    i=0
    for k in range(1,m):
        def func1(v):
            return ((x[i+1] - v)*daughter_dist(x[k]))/(x[i+1] - x[i])
        
        n[i][k]= quad(func1,x[i], x[i+1])[0]
    for i in range(1,m):
        k=i
        # the first integral is zero for diagonal terms
        def func2(v):
            return ((v-x[i-1])*daughter_dist(x[k]))/(x[i]-x[i-1])
        n[i][k]= quad(func2, x[i-1], x[i])[0]
        # terms above the main diagonal 
        for k in range(i+1,m):
            def func3(v):
                return ((x[i+1] - v)*daughter_dist(x[k]))/(x[i+1] - x[i])
            def func4(v):
                return ((v-x[i-1])*daughter_dist(x[k]))/(x[i]-x[i-1])
            
            n[i][k]= quad(func3,x[i], x[i+1])[0] + quad(func4,x[i-1], x[i])[0]

    #  eta_ijk calculation  eta(i,j,k) is contribution to ith pivot when particles on jth and kth pivot aggregate

    eta=np.zeros((m,m,m))
    for j in range(m):
        for k in  range(j,m):
            v= x[j] + x[k]
            for i in range (m-1): 
                if v> x[i]  and  v<= x[i+1] :
                    eta[i+1][j][k] = (v-x[i])/(x[i+1]-x[i]) 
                    eta[i][j][k]= (x[i+1]-v)/(x[i+1]-x[i]) 
                    break
    #  -----------------------kronecker delta matrics -----------------------------------
    dgnl=np.zeros((m,m))
    for  row in range(m):
            for col  in range(m):
                if (row == col):
                    dgnl[row][col]= 1
                else:
                    dgnl[row][col]= 0

    # ----------------------------------------kernel matrics calculation---------------------------
    a = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            a[i][j]= aggr_kernel(x[i],x[j])
       
        
    #  ---------------------ode solver for breakage and aggregation ---------------------------
    def aggregation_breakage(tspan,Nstart):
        dN_dt=np.zeros(m)
        breakage=np.zeros(m)
        aggregation= np.zeros(m)
        for i in range(m):
            #Compute the birth term
            birth = 0
            for k in range(i,m):
                birth=birth + n[i][k]*Gam[k]* Nstart[k]
            # Complete discretised PBE
            breakage[i]= birth - Gam[i] * Nstart[i]
        # Aggregation
        for i in range(m):
            birth=0
            for j in range(m):
                for k in range(j,m):
                    birth= birth +(1-0.5* dgnl[j][k]) * eta[i][j][k]* Nstart[j]* Nstart[k]*a[j][k]
            sum_N=0
            for k in range(m):
                sum_N+= Nstart[k] * a[i][k];           
            aggregation[i]= - Nstart[i]*sum_N + birth
    
        for i in range(m):
            dN_dt[i] = breakage[i] + aggregation[i]
        return dN_dt

    # --------------------------- calling the ode solver -----------------------------------
    res= solve_ivp(aggregation_breakage, tspan, Nstart)
    df = {'t':[],'volume': [],'number_density': [],'number_of_particle': [] }
    Ntot_simulated_0=[]     # for zeroth moment
    Ntot_simulated_1=[]    # for first moment
    Ntot_simulated_2=[]    # for second moment
    t= res.t
    for i in range(len(res.t)):
        s=0
        s1=0
        s2=0
        for j in range(m):
            s+= res.y[j][i]
            s1+= x[j]*res.y[j][i]
            s2+= x[j]**2*res.y[j][i]
        if (i==0):
            mu=s
            mu1=s1
            mu2=s2
        Ntot_simulated_0.append(s/mu) 
        Ntot_simulated_1.append(s1/mu1)
        Ntot_simulated_2.append(s2/mu2)
    df1 = {'time':t,'zeroth_moments': Ntot_simulated_0,'first_moments': Ntot_simulated_1,'second_moments': Ntot_simulated_2 }

    arr=list(map(float, arr.split()))
    for i in range(len(t)):
        for j in range(m):
            if t[i ] in arr:
                if j == 0:
                    df['number_density'].append(res.y[j+1][i]/width[j+1] + 0.0000023)
                    df['number_of_particle'].append(res.y[j+1][i])
                    df['t'].append(t[i])
                    df['volume'].append(x[j]) 
                else:
                    df['number_density'].append(res.y[j][i]/width[j])
                    df['number_of_particle'].append(res.y[j][i])
                    df['t'].append(t[i])
                    df['volume'].append(x[j]) 

    df2 = {'t':[],'volume': [],'number_density': [] }
    for i in range(len(t)):
        for j in range(m):
            if j == 0:
                df2['number_density'].append(res.y[j+1][i]/width[j+1] + 0.0000023)
                df2['t'].append(t[i])
                df2['volume'].append(x[j]) 
            else:
                df2['number_density'].append(res.y[j][i]/width[j])
                df2['t'].append(t[i])
                df2['volume'].append(x[j]) 

    df = pd.DataFrame(df)
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    return  Ntot_simulated_0,Ntot_simulated_1,Ntot_simulated_2,t,df,df1, df2

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Handle form submission
        v_min = request.form['minVolume']
        v_max = request.form['maxVolume']
        t_start = request.form['startTime']
        t_end = request.form['endTime']
        r = request.form['gridRatio']
        arr =request.form['numberDensity']
        breakage_kernel = request.form['breakageRate']
        daughter_distribution = request.form['daughterDistribution']
        initial_condition1 = request.form['initialCondition']
        aggregation_kernel = request.form['aggregationkernel']

        # Do something with the form data, e.g. store it in a database
        global  Ntot_simulated_0,Ntot_simulated_1,Ntot_simulated_2,t,df,df1, df2
        Ntot_simulated_0,Ntot_simulated_1,Ntot_simulated_2,t,df,df1,df2 = run_simulation(t_start, t_end, v_min, v_max, r,arr, initial_condition1, aggregation_kernel, daughter_distribution, breakage_kernel)
        return redirect(url_for('Zeroth_Moments', ))
    return render_template('form.html')


@app.route('/Zeroth_Moments')
def Zeroth_Moments(for_view=False):
    data = [
        go.Scatter(x=t, y=Ntot_simulated_0, name='numerical_0'),
    ]
    layout = go.Layout(
        title=' Zeroth Moments Distribution Curve',
        xaxis=dict(title='Time'),
        yaxis=dict(title='mu(t)/mu(0)'),
       
    )
    fig = go.Figure(data=data, layout=layout)
    chart = plot(fig, output_type='div')
    if for_view:
        return chart
    # Render the HTML template with the graph data
    return render_template('Zeroth_Moments.html', chart=chart)
    


@app.route('/First_Moments')
def First_Moments(for_view=False):
    data = [go.Scatter(x=t, y=Ntot_simulated_1, name='numerical'),]
    layout = go.Layout(
        title='First Moments Distribution Curve',
        xaxis=dict(title='Time',type='log'),
        yaxis=dict(title='mu1(t)/mu1(0)',type='log'),
    )
    fig1 = go.Figure(data=data, layout=layout)
    chart1 = plot(fig1, output_type='div')
    if for_view:
        return chart1
    return render_template('First_Moments.html', chart=chart1)

@app.route('/Second_Moments')
def Second_Moments(for_view=False):
    data = [
        go.Scatter(x=t, y=Ntot_simulated_2, name='numerical'),
    ]
    layout = go.Layout(
        title='Second Moments Distribution Curve',
        xaxis=dict(title='Time'),
        yaxis=dict(title='mu2(t)/mu2(0)'),
    )
    fig2 = go.Figure(data=data, layout=layout)
    chart2 = plot(fig2, output_type='div')
    if for_view:
        return chart2
    return render_template('Second_Moments.html', chart=chart2)


@app.route('/Density_Distribution')
def Density_Distribution(for_view=False):
    data = [go.Scatter(x=df['volume'], y=df['number_density'], name='numerical'),]
    layout = go.Layout(
        title='Number density distribution',
        xaxis=dict(title='Volume', type='log'),
        yaxis=dict(title='Number density'),
    )
    fig3 = go.Figure(data=data, layout=layout)
    chart3 = plot(fig3, output_type='div')
    if for_view:
        return chart3 
    else:
        return render_template('Density_Distribution.html', chart=chart3)
# Define route to display plots
@app.route('/display_plots')
def display_plots():
    plot1 = Zeroth_Moments(for_view=True)
    plot2 =  First_Moments(for_view=True)
    plot3 =  Second_Moments(for_view=True)
    plot4 = Density_Distribution(for_view=True)

    return render_template('plots.html',
                           plot1=plot1,
                           plot2=plot2,
                           plot3=plot3,
                           plot4=plot4)

   
@app.route('/download_data')
def download_data():
    in_memory = io.BytesIO()
    zip_file = zipfile.ZipFile(in_memory, mode='w', compression=zipfile.ZIP_DEFLATED)
    # To create CSV files, use the to_csv method of the pandas DataFrame
    zip_file.writestr('/Users/manishsaini/Downloads/M.Tech_Project/number_density_at_given_time_stamp.csv', df.to_csv(index=False, header=True))
    zip_file.writestr('/Users/manishsaini/Downloads/M.Tech_Project/moments.csv', df1.to_csv(index=False, header=True))
    zip_file.writestr('/Users/manishsaini/Downloads/M.Tech_Project/number_density.csv', df2.to_csv(index=False, header=True))


    # ...
    zip_file.close()

    # Return the ZIP file as a response to the user
    response = make_response(in_memory.getvalue())
    response.headers['Content-Type'] = 'application/zip'
    response.headers['Content-Disposition'] = 'attachment; filename=numerical_data.zip'
    return response






if __name__ == '__main__':
    app.run(debug=False)








