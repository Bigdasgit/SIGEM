
## Please note that the link prediction results in this page, are obtained by randomly removing 30% of links, while those in the SIGEM OpenReview page are obtained by randomly removing 10% of links.

**(1) GraphSAGE:** We could not utilize the original implementation of GraphSAGE since it fails on directed graphs with sink nodes, which all the directed graphs used in our paper have. Therefore, we conducted the experiments by using its improved implementation from SigMaNet [9]. We would also like to note that GraphSAGE suffers from **D4** (i.e., Limited Applicability) since it performs end-to-end training for only link prediction and node classification tasks. The results of GrashSAGE on these two tasks are as follows:

<table>
  <caption><strong>Table 1: Results of the link prediction task</strong></caption>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">CoCit</th>
      <th colspan="2">Cora</th>
      <th colspan="2">Epins</th>
      <th colspan="2">Last</th>
      <th colspan="2">Live</th>
      <th colspan="2">Pokec</th>
      <th colspan="2">VK</th>
      <th colspan="2">Google</th>      
    </tr>
    <tr>
      <th></th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>      
    </tr>
  </thead>
  <tbody>
    <tr><td><code>GraphSAGE</code></td><td>86.76</td><td>79.51</td><td>85.24</td><td>77.46</td><td>97.05</td><td>92.01</td><td>93.60</td><td>86.25</td><td>86.60</td><td>79.03</td><td>OOM</td><td>OOM</td><td>83.80</td><td>76.18</td><td>88.74</td><td>82.55</td></tr>
    <tr><td><code>SIGEM</code></td><td><b>95.98</b></td><td><b>99.32</b></td><td><b>98.27</b></td><td><b>99.50</b></td><td><b>98.32</b></td><td><b>98.22</b></td><td><b>96.57</b></td><td><b>98.09</b></td><td><b>97.92</b></td><td><b>99.43</b></td><td><b>97.09</b></td><td><b>97.98</b></td><td><b>95.85</b></td><td><b>97.50</b></td><td><b>99.28</b></td><td><b>99.78</td></tr>    
  </tbody>
</table>

<table>
  <caption><strong>Table 2: Results of the node classification task with the CoCit dataset</strong></caption>
  <thead>
    <tr>
      <th></th>
      <th>&nbsp;&nbsp;1%</th>
      <th>&nbsp;&nbsp;3%</th>
      <th>&nbsp;&nbsp;5%</th>
      <th>&nbsp;&nbsp;7%</th>
      <th>&nbsp;&nbsp;9%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GraphSAGE</code></td>
      <td>23.41</td>
      <td>25.81</td>
      <td>24.08</td>
      <td>25.08</td>
      <td>23.09</td>
    </tr>
    <tr>
      <td><code>SIGEM</code></td>
      <td><b>47.17</b></td>
      <td><b>46.44</b></td>
      <td><b>44.14</b></td>
      <td><b>43.79</b></td>
      <td><b>44.05</b></td>
    </tr>
  </tbody>
</table>

<table >
  <caption><strong>Table 3: Results of the node classification task with the Cora dataset</strong></caption>
  <thead>
    <tr>
      <th></th>
      <th>&nbsp;&nbsp;1%</th>
      <th>&nbsp;&nbsp;3%</th>
      <th>&nbsp;&nbsp;5%</th>
      <th>&nbsp;&nbsp;7%</th>
      <th>&nbsp;&nbsp;9%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GraphSAGE</code></td>
      <td>11.69</td>
      <td>12.39</td>
      <td>09.50</td>
      <td>08.88</td>
      <td>10.27</td>
    </tr>
    <tr>
      <td><code>SIGEM</code></td>
      <td><b>59.48</b></td>
      <td><b>57.70</b></td>
      <td><b>56.95</b></td>
      <td><b>57.09</b></td>
      <td><b>57.17</b></td>
    </tr>
  </tbody>
</table>

As observed in Tables 1~3, SIGEM significantly outperforms GraphSAGE on both link prediction (Table 1) and node classification (Tables 2 and 3) tasks with all evaluated datasets. The possible reason is that GraphSAGE exploits both node features and graph structure in the embedding process. However, since our datasets are attributeless, GraphSAGE uses ‘identity features’ as node attributes, which likely undermines the learning quality.

---

**(2) GATv2:** It improves GAT by modifying the order of GAT’s operations to enable a dynamic attention mechanism where the ranking of the attention scores is conditioned on the query node. However, GATv2 performs a counter-intuitive approach when handling directed graphs by transforming the asymmetric adjacency matrix to a symmetric one (e.g., please refer to line 181 at “GATv2 GitHub page”/citation2_exp.py). This simplification ignores link directions, thereby undermining the learning quality for directed graphs, where link direction conveys essential structural and semantic information [36][57]. For example, consider nodes $a$ and $b$ in our sample graph in Table 1: there exist a directed link $a \rightarrow b$ but not the reverse $b \rightarrow a$; GATv2 treats both directions as equivalent in the both learning and inferring processes.

Inspired by the aforementioned discussion, we conducted additional experiments on the link prediction task with GATv2 and a variant of it that preserves the links directions by keeping the adjacency matrix asymmetric; we call this variant GATv2-TrueDir (i.e., we did not transform the asymmetric adjacency matrix to a symmetric one). 

<table>
  <caption><strong> Table 1: Results of the link prediction task</strong></caption>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">CoCit</th>
      <th colspan="2">Cora</th>
      <th colspan="2">Epins</th>
      <th colspan="2">Last</th>
      <th colspan="2">Live</th>
      <th colspan="2">Pokec</th>
      <th colspan="2">VK</th>
      <th colspan="2">Google</th>
    </tr>
    <tr>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
      <th>AUC</th><th>PRS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GATv2</code></td>
      <td>91.00</td><td>97.91</td>
      <td>95.36</td><td>98.55</td>
      <td><b>98.48</b></td><td><b>98.39</b></td>
      <td>95.77</td><td>96.57</td>
      <td>96.55</td><td>98.32</td>
      <td>OOM</td><td>OOM</td>
      <td>93.53</td><td>92.40</td>
      <td>97.76</td><td>99.08</td>
    </tr>
    <tr>
      <td><code>GATv2-TrueDir</code></td>
      <td>86.01</td><td>92.18</td>
      <td>90.87</td><td>90.33</td>
      <td>98.11</td><td>96.32</td>
      <td>92.28</td><td>82.64</td>
      <td>96.39</td><td>98.28</td>
      <td>OOM</td><td>OOM</td>
      <td>93.40</td><td>92.95</td>
      <td>97.76</td><td>96.32</td>
    </tr>    
    <tr>
    <tr><td><code>SIGEM</code></td><td><b>95.98</b></td><td><b>99.32</b></td><td><b>98.27</b></td><td><b>99.50</b></td><td>98.32</td><td>98.22</td><td><b>96.57</b></td><td><b>98.09</b></td><td><b>97.92</b></td><td><b>99.43</b></td><td><b>97.09</b></td><td><b>97.98</b></td><td><b>95.85</b></td><td><b>97.50</b></td><td><b>99.28</b></td><td><b>99.78</td></tr>    
    </tr>
  </tbody>
</table>

(1) Undirected Graphs: As observed in Tables 1, GATv2-TrueDir and GATv2 provide nearly *identical* AUC and precision values with the both undirected datasets Live and VK. This is expected since with an undirected graph, the adjacency matrix is inherently symmetric, which makes GATv2-TrueDir and GATv2 *functionally equivalent*.

(2) Directed Graphs: GATv2 outperforms GATv2-TrueDir on directed graphs. This result is also expected since GATv2 ignores link directions; treating a directed graph incorrectly as undirected simplifies the training (embedding) and inference process, potentially at the cost of ignoring important structural information.

(3) Comparison with SIGEM: SIGEM outperforms both GATv2 and GATv2-TrueDir on both directed and undirected graphs except with the Epins dataset where GATv2 shows slightly higher AUC (98.48 vs 98.32) and precision (98.39 vs 98.22) values (possibly because GATv2 ignores link directions). The possible reason is that GATv2 may inherit the GCN’s limitations in gathering contextual information for low-degree nodes [46][59].

---

 **(3) GRACE and BGRL:**  GRACE and BGRL require node attributes; they failed to run on attributeless data (all datasets in our paper are attributeless). We tried to modify their code to use ‘identity features’ for attributeless graphs, following GraphSAGE [NIPS’17]: 
 
 <table>
  <caption><strong>Table 1: Results of the node classification task with the CoCit dataset</strong></caption>
  <thead>
    <tr>
      <th></th>
      <th>&nbsp;&nbsp;1%</th>
      <th>&nbsp;&nbsp;3%</th>
      <th>&nbsp;&nbsp;5%</th>
      <th>&nbsp;&nbsp;7%</th>
      <th>&nbsp;&nbsp;9%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GRACE</code></td>
      <td>17.46</td>
      <td>17.79</td>
      <td>18.01</td>
      <td>17.97</td>
      <td>17.47</td>
    </tr>
    <tr>
      <td><code>BGRL</code></td>
      <td>19.27</td>
      <td>17.32</td>
      <td>18.44</td>
      <td>18.72</td>
      <td>18.24</td>
    </tr>    
    <tr>
      <td><code>SIGEM</code></td>
      <td><b>47.17</b></td>
      <td><b>46.44</b></td>
      <td><b>44.14</b></td>
      <td><b>43.79</b></td>
      <td><b>44.05</b></td>
    </tr>
  </tbody>
</table>

<table >
  <caption><strong>Table 2: Results of the node classification task with the Cora dataset</strong></caption>
  <thead>
    <tr>
      <th></th>
      <th>&nbsp;&nbsp;1%</th>
      <th>&nbsp;&nbsp;3%</th>
      <th>&nbsp;&nbsp;5%</th>
      <th>&nbsp;&nbsp;7%</th>
      <th>&nbsp;&nbsp;9%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>GRACE</code></td>
      <td>14.51</td>
      <td>13.62</td>
      <td>12.34</td>
      <td>12.52</td>
      <td>12.21</td>
    </tr>    
    <tr>
      <td><code>BGRL</code></td>
      <td>15.09</td>
      <td>17.12</td>
      <td>16.22</td>
      <td>16.71</td>
      <td>16.79</td>
    </tr>
    <tr>
      <td><code>SIGEM</code></td>
      <td><b>59.48</b></td>
      <td><b>57.70</b></td>
      <td><b>56.95</b></td>
      <td><b>57.09</b></td>
      <td><b>57.17</b></td>
    </tr>
  </tbody>
</table>

 
As we observed in Tables 1 and 2, SIGEM significantly outperforms both GRACE and BGRL. The possible reason is that these methods take advantage of both node features (matrix $\mathcal{X}$) and graph structure (adjacency matrix $\mathcal{A}$) in the embedding process. However, since our datasets are attributeless, we modified their code to use ‘identity features’ as node attributes, following GraphSAGE [NIPS’17], which likely undermines their learning quality. 

 
