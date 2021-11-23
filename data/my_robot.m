clc;
clear;

deg = pi/180;
dim = 6;
dt = 0.01;
loop = 1000;
store = zeros(loop,50,50,dim*3); % (数据量，轨迹数，轨迹上的点数，参数)
for m = 1:1:loop
    d_rand = rand(1,dim);
    a_rand = rand(1,dim);
    alpha_rand = randsrc(1,dim,[-pi/2,pi/2,0]);
    m_rand = rand(1,dim) * 10;
    r_rand = rand(3,dim);
    I_rand = rand(6,dim);
    

    for i = 1:1:dim
        L(i) = Revolute('d', d_rand(1,i), 'a', a_rand(1,i), 'alpha', alpha_rand(1,i), ...
        'I', I_rand(:,i), ...
        'r', r_rand(:,i), ...
        'm', m_rand(1,i), ...
        'qlim', [-180 180]*deg );

    end
    

    robot = SerialLink(L, 'name', 'Puma 560', ...
        'manufacturer', 'Unimation');
%      
% robot.plot(zeros(dim));
     % robot.teach;
    %% 0-1范围的，如何缩放？
    for n = 1:1:50
        Q_rand =rand(1,dim) * 2*pi - pi;
        Qd_rand = rand(1,dim)*10 - 5;
        for j = 1:1:50
            torque_rand = rand(1,dim)*10 - 5;
            
            store(m,n,j,1:dim) = Q_rand;
            store(m,n,j,dim+1:dim*2) = Qd_rand;
            store(m,n,j,dim*2+1:dim*3) = torque_rand;
            
            %% 加速度
            acceleration = robot.accel(Q_rand, Qd_rand, torque_rand)';
            if numel(find(isnan(acceleration))) ~= 0
                print('error');
            end
            % [t q qd] = robot.fdyn(1, @() torque_rand, Q_rand, Qd_rand);
            Qd_rand = Qd_rand + acceleration*dt;
            Q_rand = Q_rand + Qd_rand*dt + 0.5 * acceleration * dt * dt;
            robot.plot(Q_rand)
        end

    end
end
formatOut = 'mm-dd-yy';
save(['data2' datestr(now, formatOut) '.mat'],'store') 