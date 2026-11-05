function [x_stack,u_stack]=visualizationAux(N,x,u,xInit,v_nominal,adam,ctrl,integr)

    Nreplay = N;
    x_stack = zeros(Nreplay, length(x));
    u_stack = zeros(Nreplay, length(u));

    Xk = xInit;
    for k=0:Nreplay-1
        % Integrate till the end of the interval
        R = mvnrnd(zeros(1,integr.state_dim),zeros(1,integr.state_dim),1);
        Uk = ctrl.controller(Xk, adam.optimal_param);
        Fk = integr.F('v0',v_nominal,'x0',Xk,'r',R,'p',Uk);
        Xk = Fk.xf;
        x_stack(k+1,:) = full(Xk);
        u_stack(k+1,:) = full(Uk);
    end

    if(sum(sum(x_stack>10000))>0)
        disp("this v_nominal makes the controller unstable")
        disp(v_nominal)
    end

    number_of_elements_per_columns = 4;
    number_of_sub_images  = length(Uk) + length(Xk);
    subplot_rows          = ceil(number_of_sub_images/number_of_elements_per_columns);

    input_index = 1;
    for i = 1:number_of_sub_images
        if(i <= length(Xk))
             % plotting state
            subplot(subplot_rows,number_of_elements_per_columns,i);
            grid on;hold on;
            box on; hold on;
           plot(x_stack(:,i));
           legend(['$x_' num2str(i) '$'], 'interpreter', 'latex', 'FontSize', 10);

        else
            % plotting input
            subplot(subplot_rows,number_of_elements_per_columns,i);
            grid on;hold on;
            box on; hold on;
            plot(u_stack(:, input_index));
            legend(['$u_' num2str(input_index) '$'], 'interpreter', 'latex', 'FontSize', 10);

            input_index = input_index + 1;
        end
    end

end