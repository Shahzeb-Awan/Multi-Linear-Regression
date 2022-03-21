%% Load Data
data = load('Shahzeb_Awan_data_weather.txt');
X = data(:, 1:5);
y = data(:, 6);
m = length(y);

%% ================ Part 1: Feature Normalization ================
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choosing some alpha value
alpha = 0.01;
num_iters = 1000;

% Init Theta and Running Gradient Descent 

theta =[1;44;9;2;5;1];%nice then zeros(comparitively better)

[theta, J1] = gradientDescentMulti2(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J1), J1, 'b');
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%if Humidity is 0.88 , Wind bearing degrees 141, pressure in mb is 1021.28, wind speed in km/h is 14.007
%and visibility in km is 6.0214 then Temperature should be round about :-2.22 C
Temperature=[1,(0.88-mu(1))/sigma(1),(141-mu(2))/sigma(2),(1021.28-mu(3))/sigma(3),(14.007-mu(4))/sigma(4),(6.0214-mu(5))/sigma(5)]*theta; 

% ============================================================

fprintf(['Predicted Temperature ' ...
        '(using gradient descent):\n %f\n'], Temperature);

%% ================ Part 3: Normal Equations ================
%% Analytical solution through ordinary least squares

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('Shahzeb_Awan_data_weather.txt');
X = data(:, 1:5);
y = data(:, 6);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%if Humidity is 0.88 , Wind bearing degrees 141, pressure in mb is 1021.28, wind speed in km/h is 14.007
%and visibility in km is 6.0214 then Temperature should be round about :-2.22 C

Temperature = [1,0.88	,	141	,	1021.28	,	14.007	,	6.0214]*theta;


% ============================================================

fprintf(['Predicted Temperature ' ...
         '(using normal equations):\n %f\n'], Temperature);
     
     
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-60, 100, 100);
theta1_vals = linspace(-60, 60, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j);0.002519;-0.003176;-0.179258;0.361726];
	  J_vals(i,j) = computeCostMulti(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');     


% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%my minimum cost Function Result at my Selected thetas from gradient decent:-

J=computeCostMulti(X, y, theta)