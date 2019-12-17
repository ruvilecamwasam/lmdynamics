%% LevitatingMirror1D.m 
% Simulate the levitating mirror in one dimension.


%% Run the script %%
intParams = getInternalParams();
tRange = intParams.tRange;
ss = getSteadyStates();
%    [x, px, z]
w0 = [ss.xs+2, ss.pxs,  ss.zs-1];
[t,w] = simEOM(tRange,w0,@idealEOM,@plotTimeDomainDuring);
plotTimeDomain(t,w)

function clrs = getColours(object)
    clr = crayolaColours(); %This is probably inefficient
    switch object
        case 'x'
            clrs = clr.denim;
        case 'y'
            clrs = clr.asparagus;
        case 'z'
            clrs = clr.copper;
        case 'px'
            clrs = clr.blue;
        case 'py'
            clrs = clr.green;
        case 'a'
            clrs = clr.plum;
        case 'L'
            clrs = clr.cerise;
        case 'Approx'
            clrs = clr.carnationpink;
        case 'PT'
            clrs = clr.burntorange;
        case 'XY'
            clrs = clr.denim;
    end
end


 
%% Simulation parameters %%
function params = getParams()
    params = struct('g',0.5,...
        'zeta',-30,'gamma',3e-4);
end

function steadyStates = getSteadyStates()
    params = getParams();
    g = params.g;
    zeta = params.zeta;
    
    %x_s: Stable position, x_us: Unstable position, as: Cavity steady
    %amplitude
    steadyStates = struct('xs',-g*(-zeta)+sqrt(1/g - 1),...
        'xus',-g*(-zeta)-sqrt(1/g - 1),'pxs',0,...
        'zs',-zeta*g);
end

function figs = getFigures()
    figs = struct('timeDomain',figure(1),'PhaseSpace3D',figure(2),...
        'DynamicalVariables',figure(3));
end

function intParams = getInternalParams()
%     ss = getSteadyStates();
    intParams = struct('numPoints',1e3,'xRange',5,...
        'ticTime',1,'tocTime',3,'xHalt',5,...
        'tRange',[0,1e3]);
end
%% Simulation functions %%

%Simulate a general EOM
function [t,w] = simEOM(tRange,w0,EOM,plotFunction)
    [t,w] = ode45(EOM,tRange,w0,odeset('outputfcn',plotFunction));
%      [t,y] = ode45(EOM,tRange,y0);
end

%Plot the time domain with dimensionless potential
function status = plotTimeDomainDuring(t,w,flag,varargin)
    status = 0;
    TWInitSize = 1e2;
    
    intParams = getInternalParams();
    ticTime = intParams.ticTime;
    xHalt   = intParams.xHalt;
    
    ss = getSteadyStates();
    
    persistent pTimer T W n

    switch flag
        %Set up variables for the first run
        case 'init'
            
            %Initialise arrays
            T = NaN(1,TWInitSize);
            W = NaN(3,TWInitSize);%NaN(TYInitSize,3);
            n = 1;
            
            %Timer for when to redraw plots
            pTimer = clock;
            

        otherwise
            
            %Add the new points to the array            
            numNewPoints = numel(t);
            newIndex = n+numNewPoints-1;
            %Dynamically expand if tar
            if newIndex > numel(T)
                len = numel(T);
                out(strcat('Re-allocating T,W, new len=',string(2*len)));
                T = [T, NaN(1,len)];
                W = [W, NaN(3,len)];
            end
            T(n:newIndex)   = t';
            W(:,n:newIndex) = w;
            n = n + numNewPoints;
            
            if etime(clock,pTimer) > ticTime
                out(strcat('Plotting at t=',string(t(end))));
                plotTimeDomain(T(1:n-1)',W(:,1:n-1)');
                pTimer = clock;
                
                %Halt if the x has moved too far
                if min(w(1,:)+w(3,:)) < -10
                    out(strcat('#x too large, halting simulation at t=',...
                        string(t(end))));
                    status = 1;
                end
            end
    end
end

%Plot the time domain with dimensionless potential
function plotTimeDomain(t,w)
    
    %Get the correct figure
    figs = getFigures();
    figtd = figs.timeDomain;
    figure(figtd);
    clf
    ss=getSteadyStates();
    
    %Retrieve the simulation values
    x  = w(:,1);
    px = w(:,2);
    z  = w(:,3);
    
    xT = x + z;
    pT = px;
    
    %Plot the dimensionless potential
    intParams = getInternalParams();
    numPoints = intParams.numPoints;
    xRange = intParams.xRange;
    
    xTVals = linspace(min([xT; -xRange]),max([xT; xRange]),numPoints);
    tVals = linspace(t(1),t(end),numPoints);
    [X,Y] = meshgrid(xTVals,tVals);
    Z = dimPotential(X);
    
    surf(X,Y,Z,'edgecolor','none');
    alpha(0.5);
    xlabel('$\tilde{x}+\tilde{z}$','Interpreter','Latex');
    ylabel('$\tilde{t}$','Interpreter','Latex');
    zlabel('$V(\tilde{x}+\tilde{z})$','Interpreter','Latex');
    
    %Plot the trajectory
    v = dimPotential(xT);
    
    try
        patch([xT; NaN], [t; NaN], [v; NaN], [t; NaN], 'edgecolor', 'interp');
    catch ME
%         disp('some error');
    end
    
    %Plot the 3D Phase Space
    fig3d = figs.PhaseSpace3D;
    figure(fig3d);
    clf
    
    params=getParams();
    zeta = params.zeta;
    gamma = params.gamma;
    g = params.g;

    pVals = linspace(min(pT),max(pT),numPoints);

    try
        patch([x; NaN], [px; NaN], [z; NaN], [t; NaN],...
            'edgecolor', 'interp');
        view(160,45);
        xlabel('$\tilde{x}$','Interpreter','Latex');
        ylabel('$\tilde{p}_x$','Interpreter','Latex');
        zlabel('$\tilde{z}$','Interpreter','Latex');
    catch ME
%         disp('some error');
    end
    
    %Plot the individual dynamical variables
    figdv = figs.DynamicalVariables;
    figure(figdv);
    clf;

    
    subplot(2,2,1);
    hold on
    plot(t,x,'Color',getColours('x'));
    plot(t,z,'Color',getColours('z'));
%     plot([t(1),t(end)],[ss.zs, ss.zs],':','Color',getColours('z'));
    plot(t,xT,'k--');
    hold off
    title('$\tilde{x}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(2,2,2);
    hold on
    plot(t,px,'Color',getColours('px'));
    hold off
    title('$\tilde{p}_x$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    
    subplot(2,2,3);
    
    energyT = (px.^2)/2+g*x+lightPotential(x+z);
    T=(2*pi)/sqrt(2*sqrt((1-g)*g^3));
    num=floor(10*T/(mean(diff(t))));
    energyTAvg = movmean(energyT,num);
    

    
    hold on
    plot(t,energyT,'--k');
    plot(t,energyTAvg,'k')
    hold off
    title('$\mathcal{E}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
    subplot(2,2,4);
    hold on
    dedt=diff(energyT)./diff(t);
    dedtAvg=diff(energyTAvg)./diff(t);
    
    writematrix([t, x, px, z, v],'photothermal.csv');
    
    
    %Calculate the amplitude
%     xMovMax = movmax(x,num);
%     xMovMin = movmin(x,num);
%     xMovAmp = xMovMax - xMovMin;
%     
%     gamma = params.gamma;
%     predHeating = -2*((g-1)*zeta*g^3)*gamma*(xMovAmp.^2);
%     
%     omega = sqrt(2*(g^2)*sqrt((1/g)-1));
%     kappa = gamma*(1-(omega^2)*zeta);
%     heatingRate = ((0.1^2)*zeta*(omega^6)*gamma)/(2*((kappa^2)+(omega^2)));

    heatingRate=2*gamma*((1-g)*(g^3)*zeta*(0.1^2));
    plot(t(1:end-1),dedt,'Color',getColours('x'));
    plot(t(1:end-1),dedtAvg,'Color','k');
    plot([t(1), t(end)],[heatingRate, heatingRate],'--g');
%     plot([t(1),t(end)],[predHeating, predHeating],':','Color','k');
%     plot(t(1:end-1),diff(energyT)./diff(t),'--k');
    hold off
    title('$d\tilde{\mathcal{E}}/d\tilde{t}$','Interpreter','latex');
    xlabel('$\tilde{t}$','Interpreter','latex');
    
end

% EOM for one-dimensional and no photothermal effects
function dydt = idealEOM(~,y)

    %Current dynamical variables
    cy = num2cell(y);
    [x, px, z] = cy{:};
    
    %System parameters
    params = getParams();
    g   = params.g;
    gamma = params.gamma;
    zeta = params.zeta;
    
    Pc = 1/(1+(x+z)^2);
    
    dxdt  = px;
    dpxdt = -g+Pc;
    dzdt = -gamma*(z+zeta*Pc);
    
    dydt = [dxdt; dpxdt; dzdt];
end

%% Other functions %%

%Return the dimensionless potential at a given x
function v = dimPotential(x)
    params = getParams();
    g = params.g;
    
    v = g*x - atan(x);
end

%Return the dimensionless potential at a given x
function v = lightPotential(x)
    params = getParams();
    
    v = - atan(x);
end

% Plot the dimensionless potential
function plotPotential(xMin,xMax)
    defaultXRange = 4;
    numPoints = 100;
    
    params = getParams();
    g = params.g;
    DeltaA = params.DeltaA;
    
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
    
    V = g*xRange - atan(xRange+DeltaA);
    figure(fig);
    plot(xRange,V);
end

function out(msg)
    disp(msg);
end