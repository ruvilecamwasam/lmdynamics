%% LevitatingMirror1D.m 
% Simulate the levitating mirror in one dimension.


%% Run the script %%
intParams = getInternalParams();
tRange = intParams.tRange;
ss = getSteadyStates();
%% [x0 p0 a0]
yStart = [ss.xs+0.25; ss.ps; ss.as+0.1];
y0 = [yStart(1); yStart(2); yStart(3); yStart(1); yStart(2)];
[t,y] = simEOM(tRange,y0,@idealAndApproxEOM,@plotTimeDomainDuring);

plotWavelet(t,y);

%% Simulation parameters %%
function params = getParams()
    params = struct('g',0.5,'Delta0',0,'eta',10);
end

function steadyStates = getSteadyStates()
    params = getParams();
    g = params.g;
    Delta0 = params.Delta0;
    
    %x_s: Stable position, x_us: Unstable position, as: Cavity steady
    %amplitude
    steadyStates = struct('xs',Delta0 + sqrt(1/g - 1),...
        'xus',Delta0 - sqrt(1/g - 1),'as',sqrt(g),'ps',0);
end

function figs = getFigures()
    figs = struct('timeDomain',figure(1),'PhaseSpace3D',figure(2),...
        'DynamicalVariables',figure(3),'Wavelet',figure(4));
end

function intParams = getInternalParams()
    ss = getSteadyStates();
    intParams = struct('numPoints',100,'xRange',5,...
        'ticTime',1,'tocTime',3,'xHalt',1.1*ss.xus,...
        'tRange',[0,120],'MaxWaveletPoints',1e4);
end

%% Simulation functions %%

%Simulate a general EOM
function [t,y] = simEOM(tRange,y0,EOM,plotFunction)
    [t,y] = ode45(EOM,tRange,y0,odeset('outputfcn',plotFunction));
%      [t,y] = ode45(EOM,tRange,y0);
end

%Plot the time domain with dimensionless potential
function status = plotTimeDomainDuring(t,y,flag,varargin)
    status = 0;
    
    TYInitSize = 1e2;
    
    intParams = getInternalParams();
    ticTime = intParams.ticTime;
    xHalt   = intParams.xHalt;
    
    ss = getSteadyStates();
    
    persistent pTimer T Y n

    switch flag
        %Set up variables for the first run
        case 'init'
            
            %Initialise arrays
            T = NaN(1,TYInitSize);
            Y = NaN(length(y),TYInitSize);%NaN(TYInitSize,3);
            n = 1;
            
            %Timer for when to redraw plots
            pTimer = clock;
            
            %Clear the wavelet figure from the last run
            figs = getFigures();
            figure(figs.Wavelet);
            clf;

        otherwise
            
            %Add the new points to the array            
            numNewPoints = numel(t);
            newIndex = n+numNewPoints-1;
            %Dynamically expand if tar
            if newIndex > numel(T)
                len = numel(T);
                out(strcat('Re-allocating T,Y, new len=',string(2*len)));
                T = [T, NaN(1,len)];
                Y = [Y, NaN(length(y),len)];
            end
            T(n:newIndex)   = t';
            Y(:,n:newIndex) = y;
            n = n + numNewPoints;
            
            if etime(clock,pTimer) > ticTime
                out(strcat('Plotting at t=',string(t(end))));
                plotTimeDomain(T(1:n-1)',Y(:,1:n-1)');
                pTimer = clock;
                
                %Halt if the x has moved too far
%                 if min(y(1,:)) < 2.1*ss.xus
%                     out(strcat('#x too large, halting simulation at t=',...
%                         string(t(end))));
%                     status = 1;
%                 end
            end
    end
end

%Plot the time domain with dimensionless potential
function plotTimeDomain(t,y)
    
    %Get the correct figure
    figs = getFigures();
    figtd = figs.timeDomain;
    figure(figtd);
    clf
    
    %Retrieve the simulation values
    x = y(:,1);
    p = y(:,2);
    a = y(:,3);
    
    a2 = abs(a).^2;
    
    writematrix([t, x, p, a2, angle(a)],'fullsys.csv');
    
    xa = y(:,4);
    pa = y(:,5);
    
    writematrix([t, xa, pa],'approxsys.csv');
    
    %Plot the dimensionless potential
    intParams = getInternalParams();
    numPoints = intParams.numPoints;
    xRange = intParams.xRange;
    
    xVals = linspace(min([x; -xRange]),max([x; xRange]),numPoints);
    tVals = linspace(t(1),t(end),numPoints);
    [X,Y] = meshgrid(xVals,tVals);
    Z = dimPotential(X);
    
    surf(X,Y,Z,'edgecolor','none');
    alpha(0.5);
    xlabel('$\tilde{x}$','Interpreter','Latex');
    ylabel('$\tilde{t}$','Interpreter','Latex');
    zlabel('$V(\tilde{x})$','Interpreter','Latex');
    
    %Plot the trajectory
    v = dimPotential(x);
    
    try
        patch([x; NaN], [t; NaN], [v; NaN], [t; NaN], 'edgecolor', 'interp');
    catch ME
%         disp('some error');
    end
    
    %Plot the 3D Phase Space
    fig3d = figs.PhaseSpace3D;
    figure(fig3d);
    clf
    
    
    pVals = linspace(min(p),max(p),numPoints);
    
    hold on
    [X,P] = meshgrid(xVals,pVals);
    Z = (1+X.^2).^(-1);  
    surf(X,P,Z,'edgecolor','none');
    alpha(0.5);
    
    ss = getSteadyStates();
    plot3(ss.xs,ss.ps,ss.as^2,'.k');
    plot3(ss.xus+0*pVals,pVals,ss.as^2+0*pVals,'r');
    hold off
    
    try
        patch([x; NaN], [p; NaN], [a2; NaN], [t; NaN],...
            'edgecolor', 'interp');
        view(160,45);
        xlabel('$\tilde{x}$','Interpreter','Latex');
        ylabel('$\tilde{p}$','Interpreter','Latex');
        zlabel('$|\tilde{a}|^2$','Interpreter','Latex');
    catch ME
%         disp('some error');
    end
    
    %Plot the individual dynamical variables
    figdv = figs.DynamicalVariables;
    figure(figdv);
    clf;
    
    fullSpec = 'b';
    approxSpec = 'r';
    
    subplot(3,2,1);
    hold on
    plot(t,x,fullSpec);
    plot(t,xa,approxSpec);
    hold off
    title('$\tilde{x}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(3,2,2);
    hold on
    plot(t,p,fullSpec);
    plot(t,pa,approxSpec);
    hold off
    title('$\tilde{p}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(3,2,5);
    plot(t,a2);
    title('$|\tilde{\alpha}|^2$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(3,2,6);
    plot(t,angle(a));
    title('$\phi(\tilde{\alpha})$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    ylim([-pi,pi]);
    
    subplot(3,2,3);
    energy = (p.^2)/2+dimPotential(x);
    energya = (pa.^2)/2+dimPotential(xa);
    hold on
    plot(t,energy,fullSpec);
    plot(t,energya,approxSpec);
    hold off
    title('$\tilde{\mathcal{E}}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(3,2,4);
    hold on
    denergydt=diff(energy)./diff(t);
    denergydta=diff(energya)./diff(t);
    plot(t(1:end-1),denergydt,fullSpec);
    plot(t(1:end-1),denergydta,approxSpec);
    hold off
    title('$d\tilde{\mathcal{E}}/d\tilde{t}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    writematrix([t(1:end-1),energy(1:end-1),denergydt],'energy.csv')
    writematrix([t(1:end-1),energya(1:end-1),denergydta],'energya.csv')
end

function plotWavelet(t,y)
    x = y(:,1);
    a2 = abs(y(:,3)).^2;
    
    intParams = getInternalParams();
    tRange = intParams.tRange;
    maxPoints = intParams.MaxWaveletPoints;
    
    nt = numel(t);
    T = linspace(t(1),t(end),min(nt,maxPoints));
    nT = numel(T);
    if nT < nt
        out(strcat({'Downsampling t during Wavelet transform from '},...
            string(nt),{' to '},string(maxPoints),{' points'}));
    end
    X = interp1(t,y(:,1),T);
    A2 = interp1(t,a2,T);
    
    figs = getFigures();
    figure(figs.Wavelet);
    fs = 1/(tRange(2)/numel(t));
    
    %https://au.mathworks.com/help/wavelet/ref/cwt.html#mw_f6c38b6d-dbbc-4485-8fac-9f506666abb9
    subplot(2,1,1);
    [cfs,frq] = cwt(X,fs);
    surface(T,frq,abs(cfs));
    axis tight
    shading flat
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Wavelet transform of $\tilde{x}$','Interpreter','Latex');
    set(gca,'yscale','log');
    
    subplot(2,1,2);
    [cfs,frq] = cwt(A2,fs);
    surface(T,frq,abs(cfs));
    axis tight
    shading flat
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Wavelet transform of $|\tilde{\alpha}|^2$','Interpreter','Latex');
    set(gca,'yscale','log');
%     subplot(1,2,2);
%     cwt(A2,fs);
end

% EOM for one-dimensional and no photothermal effects
function dydt = idealAndApproxEOM(~,y)

    %Current dynamical variables
    cy = num2cell(y);
    [x, p, a, xa, pa] = cy{:};
    
    %System parameters
    params = getParams();
    g   = params.g;
    eta = params.eta;
    Delta = params.Delta0;
    
    dxdt = p;
    dpdt = -g + abs(a)^2;
    dadt = eta*(1j*(Delta-x)*a-a+1);
    
    dxadt = pa;
    dpadt = -g + 1/(1+(Delta-xa)^2) +(1/eta)*(4*pa*(xa-Delta))/(1+(xa-Delta)^2)^3;
    
    dydt = [dxdt; dpdt; dadt; dxadt; dpadt];
end

%% Other functions %%

%Return the dimensionless potential at a given x
function v = dimPotential(x)
    params = getParams();
    g = params.g;
    
    v = g*x - atan(x);
end

% Plot the dimensionless potential
function plotPotential(xMin,xMax)
    defaultXRange = 4;
    numPoints = 100;
    
    params = getParams();
    g = params.g;
    
    figs = getFigures();
    fig = figs.timeDomain;
    
    switch nargin
        case 0
            xRange = linspace(-defaultXRange,defaultXRange,numPoints);
        case 1
            xRange = linspace(-xMin,xMin,numPoints);
        case 2
            xRange = linspace(xMin,xMax,numPoints);
    end
    
    V = g*xRange - atan(xRange);
    figure(fig);
    plot(xRange,V);
end

function out(msg)
    disp(msg);
end