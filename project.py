import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot 
import seaborn as sns
import math
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from tqdm import  tqdm

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://source.unsplash.com/random/?nature");
background-size: 180%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
</style>"""


st.markdown(page_bg_img, unsafe_allow_html=True)
# st.title("PBM For Breakage and Aggregation")

Title_html = """
    <style>
        .title h1{
            background:red; width:0ch; overflow: hidden;
            white-space:nowrap;animation: slide 6s steps(30) infinite alternate;
            border-right:1px solid black ;font-family:sans-serif;color:Blue; 
            font-size: 2.75rem;
          }
        @keyframes slide {
              0%{ width: 0ch; }
             50%{ width: 30ch; }
        }
    </style>  
    <div class="title">
        <h1>PBM For Breakage and Aggregation</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) 
def run_simulation(t_start, t_end, v_min, v_max, r,arr,initial_condition1, initial_condition2, aggregation_kernel, daughter_distribution, breakage_kernel):
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
    q=3
    # vi[0]=v_min
    for i in range(m_v):
        # vi[m_v-1-i]=  v_min * pow(r,N-i-1) 
        vi[i]= v_min + pow(2, (i- m_v +1)/q) *(v_max -v_min)
        # vi[i]=vi[i-1]*r
    for i in range(m):
        x[i]= (vi[i+1] + vi[i])/ 2 
        width[i]=  vi[i+1]-vi[i]  

    # ------------setting up the initial condtions // initial  number of particle on the pivots
    Nstart=np.zeros(m)
    if float(initial_condition1)!= 0:
        Nstart[m-1]= float(initial_condition1)
    else:
        def initial(y):
            return eval(initial_condition2)
        for i in range(m):
            Nstart[i]= quad(initial,vi[i],vi[i+1])[0]
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
    solver = st.text('Solver is doing calculation please wait ...')
    res= solve_ivp(aggregation_breakage, tspan, Nstart)
    solver.text('Solving is done now we will display result ...')
    df = {'t':[],'volume': [],'number_density': [],'number_of_particle': [] }
    Ntot_simulated_0=[]     # for zeroth moment
    Ntot_simulated_1=[]    # for first moment
    Ntot_simulated_2=[]    # for second moment
    analytical=[]
    t= res.t
    st.write("Time steps taken inside the solver", t)
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
        Ntot_simulated_0.append(s) 
        Ntot_simulated_1.append(s1/mu1)
        Ntot_simulated_2.append(s2/mu2)
    arr=list(map(float, arr.split()))
    # l=len(t)
    # numd= np.zeros((m, l))
    # for i in range(len(t)):
    #     for j in range(m):
    #         numd[j][i]= res.y[j][i]/width[j]
    # s = numd.sum(axis=0)       
    # st.write(s/s[0])
    for i in range(len(t)):
        for j in range(m):
            if t[i ] in arr:
                df['number_density'].append(res.y[j][i]/width[j])
                df['number_of_particle'].append(res.y[j][i])
                df['t'].append(t[i])
                df['volume'].append(x[j]) 
    st.write(df['number_density'])   
    return analytical, Ntot_simulated_0,Ntot_simulated_1,Ntot_simulated_2,t,df


# Define the user input form
placeholder1 = st.empty()
with placeholder1.form(" All Conditions",clear_on_submit=True): 
    t_start = st.text_input("Enter the start time value")
    t_end = st.text_input("Enter the end time  value")
    v_min = st.text_input("Enter the minimum volume of the grid")
    v_max = st.text_input("Enter the maximum volume of the grid")
    r = st.text_input("Enter the grid ratio value ")
    arr = st.text_input("Enter the values of time  separated by space at which you want to dispaly the number density ")
    initial_condition1 = st.text_input("Enter the initial condition in terms y if this is initial monodispersed  else 0")
    initial_condition2 = st.text_input("Enter the initial condition other than initial monodispersed in terms of y else 0")
    aggregation_kernel = st.text_input("Enter the aggregation kernel in form of y and x for sum or product kernel")
    daughter_distribution = st.text_input( "Enter the daughter distribution kernel in the form of y")
    breakage_kernel = st.text_input("Enter the breakage rate  kernel in the form of y")
    options = st.multiselect('select plots type you want to display', ['zeroth moment','number density', 'first moment', 'second moment'])
    submit = st.form_submit_button()

# Display the simulation results after the user submits the form
if submit :
    placeholder1.empty()
    st.success('Succesfully you filled all conditions')
    # Call the simulation function with user inputs
    analytical, Ntot_simulated_0,Ntot_simulated_1,Ntot_simulated_2,t,df = run_simulation(t_start, t_end, v_min, v_max, r,arr, initial_condition1, initial_condition2, aggregation_kernel, daughter_distribution, breakage_kernel)
    for i in options:
        if i == 'zeroth moment':
            fig,ax = matplotlib.pyplot.subplots(constrained_layout = True)
            ax.plot(t, Ntot_simulated_0,label='numerical_0')   # for loglog replace plot
            # ax.loglog(t, analytical,label='analytical')
            ax.plot(t, Ntot_simulated_1,label='numerical_1')
            ax.plot(t, Ntot_simulated_2,label='numerical_2')
            ax.set_xlabel('time')
            ax.set_ylabel('N(t)/N(0)')
            matplotlib.pyplot.legend()
            matplotlib.pyplot.xlim(0,10)
            matplotlib.pyplot.ylim(0,12)
            ax.set_title('moments distribution curves')
            st.pyplot(fig)
        elif i=='first moment':
            fig1,ax1 = matplotlib.pyplot.subplots(constrained_layout = True)
            ax1.plot(t, Ntot_simulated_1,label='numerical')
            ax1.set_xlabel('time')
            ax1.set_ylabel('mu1(t)/mu1(0)')
            matplotlib.pyplot.ylim(0,10)
            matplotlib.pyplot.ylim(0,12)
            matplotlib.pyplot.legend()
            ax1.set_title('first moments distribution curve ')
            st.pyplot(fig1)
        elif i=='second moment':
            fig2,ax2 = matplotlib.pyplot.subplots(constrained_layout = True)
            ax2.plot(t, Ntot_simulated_2,label='numerical')
            ax2.set_xlabel('time')
            ax2.set_ylabel('mu2(t)/mu2(0)')
            matplotlib.pyplot.legend()
            matplotlib.pyplot.xlim(0,10)
            matplotlib.pyplot.ylim(0,12)
            ax2.set_title('second  moments distribution curve ')
            st.pyplot(fig2)
        elif i == 'number density':
            fig3 = matplotlib.pyplot.figure()
            sns.scatterplot(data=pd.DataFrame(df),x= "volume",y= "number_density", hue="t")        
            matplotlib.pyplot.xscale('log')
            # matplotlib.pyplot.yscale('log')
            matplotlib.pyplot.xlim(1e-6,1e+2)
            matplotlib.pyplot.ylim(0, 140)
            matplotlib.pyplot.title("Number density distribution ")
            st.pyplot(fig3)
        else:
            st.write("Select  the valid the choice for display the data")
