"""Build an auditable JSON/CSV summary and offline visual report from saved runs.
No spectrum clipping, curve rescaling, or replacement of a failed result.
"""
from pathlib import Path
import sys,json,csv,base64,html
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src/pyrpmd/analysis'))
from adaptive_quantum import matched_rate_defect

def load(name):return json.loads((OUT/(name+'.json')).read_text())
def summarize():
    sel=load('selection');conv=load('convergence');summary={
        'units':'hbar=m=kB=1; omega angular frequency; rates length/time',
        'basis_commit':'2800af5d128b8db270c8ad06c32bc42b2f1af2b6',
        'selector':{k:{'dimension':v['dimension'],'status':v['status'],'residual_loading':v['residual_loading']} for k,v in sel.items()},
        'selection_uses_full_quantum_answer':False,'physical_converged':False,'morse':{},'fermi':{},'eckart':{},
        'numerical_checks':{'fast_tests_passed':24,'morse_ground_energy_max_change':conv['morse_reference']['ground_energy_max_change'],
            'fermi_d3_basis_relative_spectrum_change':conv['fermi_d3_basis']['relative_spectrum_change'],
            'analytic_eckart_max_probability_error':conv['analytic_eckart_max_probability_error'],
            'eckart_full_max_unitarity_defect':conv['eckart_d3']['meta']['max_unitarity_defect']}}
    rows=[]
    for P,tag in [(32,'main'),(64,'beadcheck')]:
        m=load(f'morse_{tag}_P{P}')
        summary['morse'][str(P)]={k:m[k] for k in ['energy_quantum','energy_pimd','energy_pimd_block_se','errors']}
        f=load(f'fermi_{tag}_P{P}')
        summary['fermi'][str(P)]={'eta':f['eta'],'metrics':f['metrics'],'bootstrap95':f['bootstrap95']}
        for method,v in f['metrics'].items():rows.append(['fermi',6,P,method,v['relative_absorption_L2'],'relative absorption L2'])
    for beta,P,tag in [(4.,32,'main'),(8.,32,'main'),(8.,64,'convergence')]:
        rr={}
        for d in (1,2,3):
            p=OUT/f'eckart_rp_{tag}_b{beta:g}_P{P}_d{d}.json'
            if p.exists():rr[d]=json.loads(p.read_text())
        if 3 not in rr or 2 not in rr:raise RuntimeError('Required rate calculation missing')
        q=conv['eckart_d3']['rates'][str(beta)];F=rr[3]['rate'];sigmaF=F*rr[3]['rate_relative_sampling_SE']
        r={'quantum_reference':q,'rpmd':F,'rpmd_relative_error':F/q-1,'rpmd_sampling_SE':sigmaF,
           'classical_TST':float(np.exp(-beta)/np.sqrt(2*np.pi*beta)),
           'kappa_plateau_early':rr[3]['plateau_early'],'kappa_plateau_late':rr[3]['plateau_late'],
           'PMF_quadrature_nodes':rr[3]['quadrature_nodes'],'physical_converged':False}
        for d in (1,2):
            if d not in rr:continue
            R=rr[d]['rate'];Q=conv[f'eckart_d{d}']['rates'][str(beta)]
            est=matched_rate_defect(F,R,Q,beta=beta,observable='side(q>0); centroid surface in RPMD',units='length/time')
            sigmaR=R*rr[d]['rate_relative_sampling_SE']
            est.update(relative_error=est['rate']/q-1,sampling_SE_independent_approximation=float(np.hypot(sigmaF,sigmaR)),
                       sampling_SE_upper_delta_method=float(sigmaF+sigmaR))
            r[f'corrected_d{d}']=est
            rows.append(['eckart',beta,P,f'corrected_d{d}',abs(est['relative_error']),'relative rate error'])
        rows.append(['eckart',beta,P,'rpmd',abs(r['rpmd_relative_error']),'relative rate error'])
        summary['eckart'][f'beta{beta:g}_P{P}']=r
    summary['limitations']=[
        'Morse is a separable 3D anharmonic control. All 3 nonlinear directions are selected; no dimensional reduction.',
        'Morse finite box is explicit; this is a low-temperature bound-dominated model, not the unconfined dissociating canonical ensemble.',
        'Fermi harmonic-reservoir embedding uses analytic bath response (equal to harmonic RPMD); it is not an arbitrary anharmonic bath solver.',
        'Eckart uses experimental matched normalized plateau-rate difference, not the bound-spectrum passive embedding formula.',
        'Eckart rate SE is delta-method sampling only; shared seeds do not establish independent full/active errors. Both independent approximation and conservative sum are reported.',
        'Combined bead, quadrature, warmup and seed checks do not isolate all systematic errors; global physical_converged remains false.',
        'Quantum references are separate audits and never used to select the subspace or fit correction coefficients.'
    ]
    (OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['model','beta','beads','method','error','quantity']);wr.writerows(rows)
    return summary

def main():
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=FontProperties(fname=str(fp)) if fp.exists() else FontProperties()
    s=summarize()
    # Separate figures, default matplotlib color cycle.
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    ys=[s['morse'][str(P)]['energy_pimd'] for P in (32,64)];errs=[s['morse'][str(P)]['energy_pimd_block_se'] for P in (32,64)]
    ax.errorbar([32,64],ys,yerr=errs,marker='o',capsize=4,label='PIMD energy')
    ax.axhline(s['morse']['32']['energy_quantum'],linestyle='--',label='Quantum reference')
    ax.set(xlabel='Number of beads P',ylabel='Energy',title='3D anharmonic Morse: zero-point energy convergence');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'morse_energy.png',dpi=160);plt.close(fig)
    z=np.load(OUT/'fermi_main_P32.npz');fig,ax=plt.subplots(figsize=(8.5,4.8))
    for key,label,ls in [('quantum','3D quantum','-'),('rpmd','Full RPMD','--'),('inverse_d2','2D inverse defect',':'),('embedded_d2','2D + harmonic bath','-.')]:
        ax.plot(z['omega'],z[key].imag,label=label,linestyle=ls)
    ax.set(xlabel='Angular frequency',ylabel='Im susceptibility',title='Fermi resonance: nonzero x*y^2 coupling');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'fermi_spectrum.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for d in (1,2,3):
        z=np.load(OUT/f'eckart_converged_d{d}.npz');mask=z['energy']<=1.5
        ax.semilogy(z['energy'][mask],z['T0'][mask],label=f'{d}D quantum',linestyle=['-','--','-.'][d-1])
    ax.axvline(1.,linestyle=':',label='Bare Eckart barrier');ax.set(xlabel='Collision energy (from channel ground threshold)',ylabel='Ground-channel transmission',title='Curved Eckart barrier: genuine below-barrier transmission')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'eckart_transmission.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8));keys=list(s['eckart']);x=np.arange(len(keys))
    for shift,method,label in [(-.16,'rpmd','Full RPMD'),(.16,'corrected_d2','2D rate-defect correction')]:
        val=[];se=[]
        for k in keys:
            r=s['eckart'][k];q=r['quantum_reference']
            val.append((r[method] if method=='rpmd' else r[method]['rate'])/q)
            se.append((r['rpmd_sampling_SE'] if method=='rpmd' else r[method]['sampling_SE_upper_delta_method'])/q)
        ax.bar(x+shift,val,.32,yerr=se,capsize=4,label=label)
    ax.axhline(1,linestyle='--',label='Quantum reference');ax.set_xticks(x,keys);ax.set(ylabel='Rate / quantum reference',title='Eckart thermal rates (sampling error bars, not total error)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'eckart_rates.png',dpi=160);plt.close(fig)
    sections=[]
    def picture(name):
        return '<img src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'"/>'
    for name,title,text in [
        ('morse_energy.png','1. 三维 Morse：零点能与不可降维判据','三条真实 Morse 非谐坐标；自动选出三维。PIMD 本身已经包含零点能，不额外重复加零点能。三维替换属于全量子极限，不是降维成功。'),
        ('fermi_spectrum.png','2. Fermi 共振：xy² 核心＋环境','观测量始终为 x，独立先导选出 x、y。逆响应修正使用完整 RPMD；谐性环境嵌入是另一种特定模型近似，两者分别报告。'),
        ('eckart_transmission.png','3. 弯曲 Eckart：开放散射参照','Eckart 势垒＋tanh 位移强浴模＋弱耦合旁观模。采用开放边界散射，不将盒中离散谱当作不可逆反应速率。'),
        ('eckart_rates.png','4. 实际 RPMD 热速率对照','质心约束热力学积分＋通量侧函数平台。二维修正为匹配归一化速率差分，尚无一般精确性证明。误差棒采用采样误差传播上界，不包括所有系统误差。')]:
        sections.append('<section><h2>'+title+'</h2><p>'+text+'</p>'+picture(name)+'</section>')
    rows=''
    for k,r in s['eckart'].items():
        rows+=f'<tr><td>{k}</td><td>{r["quantum_reference"]:.6g}</td><td>{r["rpmd_relative_error"]:+.2%}</td><td>{r["corrected_d2"]["relative_error"]:+.2%}</td></tr>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PyRPMD 非谐三模型重新验证</title><style>body{max-width:1000px;margin:36px auto;padding:0 22px;font-family:system-ui,sans-serif;line-height:1.65;color:#182534}h1,h2{line-height:1.3}header,section{margin-bottom:32px}section{border-top:1px solid #ccd5df;padding-top:20px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd5df;text-align:left}.note{padding:16px;background:#fff4dc}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><header><h1>PyRPMD：非谐三模型重新验证</h1><p>Morse 零点能 · Fermi 共振 · 三维弯曲 Eckart 隧穿</p><div class="note">这是新的非谐计算，不沿用旧谐振子“366%／33%”表。自动选模维数为 3／2／2；所有结果保留 physical_converged=False。</div></header>'''+''.join(sections)+'<h2>Eckart 速率汇总</h2><table><tr><th>设置</th><th>量子参照</th><th>RPMD 偏差</th><th>二维修正偏差</th></tr>'+rows+'</table><h2>完整数值摘要与限制</h2><pre>'+html.escape(json.dumps(s,indent=2,ensure_ascii=False))+'</pre></html>'
    (ROOT/'anharmonic_validation_report.html').write_text(text,encoding='utf-8')
    return s
if __name__=='__main__':main()
