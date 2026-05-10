import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import rasterio
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

print("🚀 GeoAI Evacuation Agent Başlatılıyor...")

# ==========================================
# 1. VERİ YÜKLEME
# ==========================================
print("📂 Veriler yükleniyor (EPSG:5254)...")

highways = gpd.read_file("kadikoy_highways_fixed.geojson")

assembly_raw = pd.read_excel("depremToplanmaAlani.xlsx")
lons = assembly_raw['coordinates'].iloc[::2].reset_index(drop=True)
lats = assembly_raw['coordinates'].iloc[1::2].reset_index(drop=True)
names = assembly_raw['TOPLANMA_ALAN_AD'].dropna().reset_index(drop=True)

assembly_areas = gpd.GeoDataFrame(
    {'name': names}, 
    geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)], 
    crs="EPSG:4326"
).to_crs(epsg=5254)

pois = gpd.read_file("poi.geojson").to_crs(epsg=5254)

# ==========================================
# 2. İNTERAKTİF KONUM SEÇİMİ (Sabit Boyutlu Ekran)
# ==========================================
print("🗺️ Lütfen açılan haritada bulunduğunuz konumu seçin.")

# Pencere boyutunu ekranın ~%60'ı olacak şekilde ayarlıyoruz
fig_select, ax_select = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')

minx, miny, maxx, maxy = highways.total_bounds
ax_select.set_xlim(minx, maxx)
ax_select.set_ylim(miny, maxy)

# Ara sokakları net görmek için vektör yollar
highways.plot(ax=ax_select, color='#34495e', linewidth=0.8, alpha=0.5, zorder=1)

try:
    import contextily as ctx
    ctx.add_basemap(ax_select, crs=highways.crs.to_string(), source=ctx.providers.CartoDB.Positron, zoom=14, zorder=0)
except Exception:
    pass

plt.title(" Bulunduğunuz konuma TIKLAYIN", fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
plt.axis('off')
plt.tight_layout()

clicked_points = plt.ginput(n=1, timeout=0, show_clicks=True)
plt.close(fig_select)

if not clicked_points:
    print("❌ Seçim yapılmadı, program kapatılıyor.")
    exit()

click_x, click_y = clicked_points[0]
user_point = Point(click_x, click_y)

# ==========================================
# 3. GRAF İNŞASI, DEM EĞİM HESABI VE KISITLAR
# ==========================================
print("🧠 Ajan karar matrisini hesaplıyor...")
G = nx.Graph()

with rasterio.open("DEM_kadikoy.tif") as dem:
    for _, row in highways.iterrows():
        geom = list(row.geometry.geoms)[0] if row.geometry.geom_type == 'MultiLineString' else row.geometry
        if geom.is_empty: continue
        
        coords = list(geom.coords)
        sim_width = row.get('sim_width', 5.0)
        h_type = row.get('highway', 'unclassified')
        safety_weight = 1.0 if h_type in ['pedestrian', 'footway', 'cycleway'] else 0.6
        
        for i in range(len(coords) - 1):
            u = coords[i]
            v = coords[i+1]
            seg_geom = LineString([u, v])
            seg_len = seg_geom.length
            
            try:
                z1 = next(dem.sample([(u[0], u[1])]))[0]
                z2 = next(dem.sample([(v[0], v[1])]))[0]
                slope = (abs(z2 - z1) / seg_len) * 100 if seg_len > 0 else 0
            except:
                slope = 0 
                
            if sim_width < 4.0 or slope > 10:
                seg_cost = seg_len * 1000  
            else:
                seg_cost = (seg_len / safety_weight) * (0.5 if h_type in ['footway', 'pedestrian'] else 1.0)
            
            street_name = row.get('name', 'İsimsiz Sokak/Geçit')
            G.add_edge(u, v, weight=seg_cost, length=seg_len, geometry=seg_geom, name=street_name)

largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

node_coords = list(G.nodes)
tree = cKDTree(node_coords)
source_node = node_coords[tree.query((user_point.x, user_point.y))[1]]

# ==========================================
# 4. 3 ALTERNATİF ROTA BULMA (1500m Kısıtı)
# ==========================================
print("🔄 1.5 km kısıtı dikkate alınarak alternatif rotalar aranıyor...")
routes = []

for _, area in assembly_areas.iterrows():
    target_node = node_coords[tree.query((area.geometry.x, area.geometry.y))[1]]
    try:
        path = nx.shortest_path(G, source=source_node, target=target_node, weight='weight')
        path_len = nx.path_weight(G, path, weight='length')
        cost = nx.path_weight(G, path, weight='weight')
        
        if path_len <= 1500:
            routes.append({'path': path, 'length': path_len, 'target': area['name'], 'cost': cost, 'target_geom': area.geometry})
    except nx.NetworkXNoPath:
        continue

routes = sorted(routes, key=lambda x: x['cost'])[:3]

if not routes:
    print("❌ 1.5 km yürüme mesafesinde güvenli toplanma alanı bulunamadı!")
    exit()

# ==========================================
# 5. DOSYALAMA VE RAPORLAMA SİSTEMİ
# ==========================================
output_folder = "Tahliye_Plani"
os.makedirs(output_folder, exist_ok=True)
print(f"\n📁 Raporlar '{output_folder}' klasörüne kaydediliyor...")

for idx, route_dict in enumerate(routes):
    r_num = idx + 1
    path_nodes = route_dict['path']
    target_name = route_dict['target']
    
    # --- A. METİN TARİFİ OLUŞTURMA (.txt) ---
    txt_filename = os.path.join(output_folder, f"Rota_{r_num}_Tarif.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"🗺️ Rota {r_num} Yönlendirme Talimatları\n")
        f.write(f"Hedef: {target_name}\n")
        f.write("="*50 + "\n")

        current_street = G[path_nodes[0]][path_nodes[1]]['name']
        segment_length = 0
        
        f.write("📍 Bulunduğunuz noktadan sokağa çıkış yapın.\n")

        for i in range(len(path_nodes)-1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            edge_data = G[u][v]
            
            next_street = edge_data['name']
            segment_length += edge_data['length']
            
            if current_street != next_street or i == len(path_nodes)-2:
                point_geom = Point(v)
                nearby_pois = pois[pois.geometry.distance(point_geom) < 50] 
                
                poi_text = ""
                if not nearby_pois.empty:
                    row_poi = nearby_pois.iloc[0]
                    poi_name = row_poi.get('name') or row_poi.get('amenity') or row_poi.get('shop') or "belirgin yapı"
                    poi_text = f" (Köşedeki '{poi_name}' konumunu referans alın)"
                    
                f.write(f"🚶 '{current_street}' boyunca yaklaşık {int(segment_length)} metre ilerleyin.\n")
                if current_street != next_street:
                    f.write(f"↪️ '{next_street}' yönüne dönün.{poi_text}\n")
                    
                current_street = next_street
                segment_length = 0
                
        f.write(f"\n📍 Sokağın sonuna ulaştığınızda '{target_name}' toplanma alanı doğrudan yakınınızda olacaktır.\n")
        f.write("✅ Geçmiş olsun.\n")

    # --- B. HARİTA GÖRSELİ OLUŞTURMA (.png) ---
    # GÖRSEL KAYMA OLMAMASI İÇİN ÇİZİMİ WEB MERCATOR (EPSG:3857) İLE YAPIYORUZ
    plot_crs = "EPSG:3857"
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')
    
    # Altlık verileri çizim projeksiyonuna çevirip çizdiriyoruz
    highways.to_crs(plot_crs).plot(ax=ax, color='#bdc3c7', linewidth=0.8, alpha=0.5, zorder=1)
    assembly_areas.to_crs(plot_crs).plot(ax=ax, color='#27ae60', markersize=100, alpha=0.9, zorder=2, edgecolors='white', linewidth=1.5)
    
    # Kullanıcı noktasını çevir
    user_pt_plot = gpd.GeoSeries([user_point], crs="EPSG:5254").to_crs(plot_crs).iloc[0]
    ax.scatter(user_pt_plot.x, user_pt_plot.y, color='#2980b9', s=150, label='Başlangıç Konumunuz', zorder=6, edgecolors='white', linewidth=2)

    # Ana rotayı çiz
    route_geoms = [G[path_nodes[i]][path_nodes[i+1]]['geometry'] for i in range(len(path_nodes)-1)]
    route_gdf = gpd.GeoSeries(route_geoms, crs="EPSG:5254").to_crs(plot_crs)
    route_gdf.plot(ax=ax, color='#e74c3c', linewidth=6, label=f"Tahliye Rotası ({int(route_dict['length'])}m)", zorder=4)

    # 1. Başlangıç noktasından ilk düğüme bağlayan kesik çizgi
    first_node_pt = Point(path_nodes[0])
    start_gap_line = LineString([user_point, first_node_pt])
    gpd.GeoSeries([start_gap_line], crs="EPSG:5254").to_crs(plot_crs).plot(ax=ax, color='#8e44ad', linewidth=3, linestyle=':', zorder=5, label='Konumdan Yola Çıkış')

    # 2. Son düğümden hedefe bağlayan kesik çizgi
    target_pt = route_dict['target_geom']
    last_node_pt = Point(path_nodes[-1])
    end_gap_line = LineString([last_node_pt, target_pt])
    gpd.GeoSeries([end_gap_line], crs="EPSG:5254").to_crs(plot_crs).plot(ax=ax, color='#e74c3c', linewidth=3, linestyle='--', zorder=3)

    minx, miny, maxx, maxy = route_gdf.total_bounds
    buffer = 250 
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    plt.title(f"Rota {r_num}: {target_name}\nToplam Yürüme Mesafesi: {int(route_dict['length'])}m", fontsize=16, fontweight='bold', pad=15)
    
    try:
        # Contextily artık kendi doğal formatında (3857) çalışacağı için kayma yapmayacak
        ctx.add_basemap(ax, crs=plot_crs, source=ctx.providers.CartoDB.Positron, zoom=15, zorder=0)
    except Exception:
        pass

    plt.legend(loc='lower right', frameon=True, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    
    png_filename = os.path.join(output_folder, f"Rota_{r_num}_Harita.png")
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close(fig) 

print(f"✅ İşlem Tamamlandı! Rotalar ve tarifler '{output_folder}' klasörüne kaydedildi.")