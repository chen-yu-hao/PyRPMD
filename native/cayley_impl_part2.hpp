                    spectral_pair(real, imag, k);
                fourier->transform_batched_serial(real, imag, dims, true);
                std::copy(real, real + size, q);
                std::copy(imag, imag + size, p);
                return;
            }
            #pragma omp parallel num_threads(threads)
            {
                #pragma omp for schedule(static)
                for (Py_ssize_t index = 0; index < size; ++index) {
                    real[index] = q[index];
                    imag[index] = p[index];
                }
                fourier->transform_batched_team(real, imag, dims);
                #pragma omp for schedule(static)
                for (Py_ssize_t k = 0; k < pairs; ++k)
                    spectral_pair(real, imag, k);
                fourier->transform_batched_team(real, imag, dims, true);
                #pragma omp for schedule(static)
                for (Py_ssize_t index = 0; index < size; ++index) {
                    q[index] = real[index];
                    p[index] = imag[index];
                }
            }
            return;
        }
        // For the Cayley map, transpose one small coordinate tile into the
        // existing [coordinate, bead] workspace before solving. The tile keeps
        // all bead values hot while q/p are read and written contiguously in
        // their public [bead, atom, xyz] layout. Arithmetic for each coordinate
        // is unchanged, preserving bitwise results across thread counts.
        if (scheme == 0 && n > 1 && dims >= 24) {
            constexpr Py_ssize_t tile = 48;
            const Py_ssize_t blocks = (dims + tile - 1) / tile;
            #pragma omp parallel for num_threads(threads) schedule(static) if(threads>1)
            for (Py_ssize_t block = 0; block < blocks; ++block) {
                const Py_ssize_t begin = block * tile;
                const Py_ssize_t end = std::min(dims, begin + tile);
                for (Py_ssize_t k = 0; k < n; ++k) {
                    const Py_ssize_t offset = k * dims;
                    for (Py_ssize_t j = begin; j < end; ++j)
                        workspace[j*n+k] = q[offset+j] + .5*dt*p[offset+j]*inverse_coordinate_mass[j];
                }
                for (Py_ssize_t j = begin; j < end; ++j) {
                    double* x = workspace + j*n;
                    if (n == 2) {
                        const double a=x[0],b=x[1],den=1.+4.*alpha;
                        x[0]=((1.+2.*alpha)*a+2.*alpha*b)/den;
                        x[1]=(2.*alpha*a+(1.+2.*alpha)*b)/den;
                    } else {
                        solve(x);
                        const double c1=si[0]*x[n-1]+si[1]*x[0];
                        const double c2=si[2]*x[n-1]+si[3]*x[0];
                        for(Py_ssize_t k=0;k<n;k++)x[k]-=z1[k]*c1+z2[k]*c2;
                    }
                }
                for (Py_ssize_t k = 0; k < n; ++k) {
                    const Py_ssize_t offset = k * dims;
                    for (Py_ssize_t j = begin; j < end; ++j) {
                        const double* x = workspace + j*n;
                        q[offset+j]=2.*x[k]-q[offset+j];
                        p[offset+j]-=dt*coordinate_mass[j]*omega2*
                            (2.*x[k]-x[(k+n-1)%n]-x[(k+1)%n]);
                    }
                }
            }
            return;
        }
        #pragma omp parallel for num_threads(threads) schedule(static) if(threads>1)
        for (Py_ssize_t j=0;j<dims;j++) {
            const double m=coordinate_mass[j];
            if(n==1){ q[j]+=dt*p[j]*inverse_coordinate_mass[j]; continue; }
            if(scheme==1){
                double* qnew=workspace+2*j*n;
                double* pnew=qnew+n;
                for(Py_ssize_t b=0;b<n;++b){
                    double x=0.,v=0.;
                    for(Py_ssize_t k=0;k<n;++k){
                        const Py_ssize_t d=(b+n-k)%n, index=k*dims+j;
                        x+=exact_a[d]*q[index]+exact_b[d]*p[index]/m;
                        v+=exact_a[d]*p[index]-exact_c[d]*m*q[index];
                    }
                    qnew[b]=x;pnew[b]=v;
                }
                for(Py_ssize_t b=0;b<n;++b){q[b*dims+j]=qnew[b];p[b*dims+j]=pnew[b];}
                continue;
            }
            if(scheme==2){
                double* qnew=workspace+j*n;
                const double factor=.5*dt*m*omega2;
                for(Py_ssize_t b=0;b<n;++b){
                    const Py_ssize_t index=b*dims+j;
                    p[index]-=factor*(2.*q[index]-q[((b+n-1)%n)*dims+j]-q[((b+1)%n)*dims+j]);
                    qnew[b]=q[index]+dt*p[index]/m;
                }
                for(Py_ssize_t b=0;b<n;++b){
                    const Py_ssize_t index=b*dims+j;
                    q[index]=qnew[b];
                    p[index]-=factor*(2.*qnew[b]-qnew[(b+n-1)%n]-qnew[(b+1)%n]);
                }
                continue;
            }
            double* x=workspace+j*n;
            for(Py_ssize_t k=0;k<n;k++) x[k]=q[k*dims+j]+.5*dt*p[k*dims+j]/m;
            if(n==2){
                const double a=x[0],b=x[1],den=1.+4.*alpha;
                x[0]=((1.+2.*alpha)*a+2.*alpha*b)/den;
                x[1]=(2.*alpha*a+(1.+2.*alpha)*b)/den;
            }else{
                solve(x);
                const double c1=si[0]*x[n-1]+si[1]*x[0];
                const double c2=si[2]*x[n-1]+si[3]*x[0];
                for(Py_ssize_t k=0;k<n;k++)x[k]-=z1[k]*c1+z2[k]*c2;
            }
            for(Py_ssize_t k=0;k<n;k++) {
                q[k*dims+j]=2.*x[k]-q[k*dims+j];
                p[k*dims+j]-=dt*m*omega2*(2.*x[k]-x[(k+n-1)%n]-x[(k+1)%n]);
            }
        }
    }
    void sample_free(double* q,double* p,const double* noise,const double* decay,
                     const double* amplitude,const unsigned char* fixed,double* work) const {
        const Py_ssize_t n=beads,dims=nd;
        const double scale0=std::sqrt(static_cast<double>(n)),scale=std::sqrt(.5*n);
        #pragma omp parallel for num_threads(threads) schedule(static) if(threads>1)
        for(Py_ssize_t j=0;j<dims;++j){
            if(fixed[j/3])continue;
            const double m=coordinate_mass[j];
            double* qr=work+6*j*n;double* qi=qr+n;double* pr=qi+n;double* pi=pr+n;double* temporary=pi+n;
            for(Py_ssize_t b=0;b<n;++b){qr[b]=q[b*dims+j];pr[b]=p[b*dims+j];qi[b]=pi[b]=0.;}
            fourier->forward_two_real(qr,qi,pr,pi,temporary);
            for(Py_ssize_t k=0;k<n;++k){
                const double xr=qr[k],xi=qi[k],vr=pr[k],vi=pi[k];
                double x=half_a[k]*xr+half_b[k]*vr/m,y=half_a[k]*xi+half_b[k]*vi/m;
                double u=half_a[k]*vr-half_c[k]*m*xr,v=half_a[k]*vi-half_c[k]*m*xi;
                Py_ssize_t mode;double nr,ni;
                if(k==0){mode=0;nr=scale0*noise[j];ni=0.;}
                else if(n%2==0 && k==n/2){mode=n-1;nr=scale0*noise[(n-1)*dims+j];ni=0.;}
                else{
                    const Py_ssize_t frequency=std::min(k,n-k);mode=2*frequency-1;
                    nr=scale*noise[mode*dims+j];ni=(k<n-k?-scale:scale)*noise[(mode+1)*dims+j];
                }
                const Py_ssize_t sine_mode=(k==0 || (n%2==0 && k==n/2)) ? mode : mode+1;
                u=decay[mode]*u+amplitude[mode]*nr;
                v=decay[sine_mode]*v+amplitude[sine_mode]*ni;
                qr[k]=half_a[k]*x+half_b[k]*u/m;qi[k]=half_a[k]*y+half_b[k]*v/m;
                pr[k]=half_a[k]*u-half_c[k]*m*x;pi[k]=half_a[k]*v-half_c[k]*m*y;
            }
            fourier->inverse_two_real(qr,qi,pr,pi,temporary);
            for(Py_ssize_t b=0;b<n;++b){q[b*dims+j]=qr[b];p[b*dims+j]=pr[b];}
        }
    }
};
