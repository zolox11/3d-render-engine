#version 330 core

in vec2 v_uv;
in vec3 v_world_pos;
in mat3 TBN;
in vec3 v_normal;

out vec4 f_color;

// --- TOGGLE ---
uniform bool u_is_terrain; // Set to true for terrain, false for GLBs/Cubes
uniform bool u_use_color; // Set to true to use solid color, false to use texture
uniform vec3 u_color; // Solid color when u_use_color is true

// --- STANDARD TEXTURES ---
uniform sampler2D texture0; // Albedo for regular objects
uniform sampler2D texture1; // MRA (Metallic, Roughness, AO)
uniform sampler2D texture2; // Normal Map
uniform sampler2D texture3; // Emissive or extra
uniform sampler2D shadowMap;

// --- TERRAIN TEXTURES ---
uniform sampler2D tex_grass;
uniform sampler2D tex_rock;
uniform sampler2D tex_sand;
uniform sampler2D tex_snow;
uniform vec3 u_grass_adj;

uniform sampler2D detailTex;
uniform sampler2D noiseTex;

// --- CAMERA & LIGHTING ---
uniform vec3 cam_pos;
uniform mat4 lightSpaceMatrix;

// --- TERRAIN PARAMS ---
uniform float u_grass_min_height;
uniform vec3 u_grass_tint = vec3(1.0);
uniform float u_grass_max_height;
uniform float u_rock_slope_start;
uniform float u_rock_slope_end;
uniform float u_snow_height;

uniform float u_noise_strength;
uniform float u_detail_strength;
uniform float u_macro_tiling;
uniform float u_detail_tiling;
uniform float u_blend_sharpness;

#define MAX_LIGHTS 8
#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1

struct Light {
    int type;
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
};

uniform int light_count;
uniform Light lights[MAX_LIGHTS];

const float PI = 3.14159265359;
const float SHADOW_BIAS = 0.005;

// ---------------- RANDOM ----------------
float random(vec3 seed) {
    return fract(sin(dot(seed, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

// ---------------- SHADOW ----------------
float ShadowCalculation(vec4 fragPosLightSpace, vec3 N, vec3 L) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;

    float bias = max(SHADOW_BIAS * (1.0 - dot(N, L)), 0.0005);

    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

    for(int x = -2; x <= 2; x++) {
        for(int y = -2; y <= 2; y++) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (projCoords.z - bias > pcfDepth) ? 1.0 : 0.0;
        }
    }

    return shadow / 25.0;
}

// ---------------- PBR MATH ----------------
vec3 ACESToneMapping(vec3 x) {
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float remapRoughness(float r) {
    return r * r;
}

// ---------------- TRIPLANAR ----------------
vec3 triplanar(sampler2D tex, vec3 pos, vec3 normal, float scale) {
    vec3 blend = abs(normal);
    blend /= (blend.x + blend.y + blend.z);

    vec2 xUV = pos.yz * scale;
    vec2 yUV = pos.xz * scale;
    vec2 zUV = pos.xy * scale;

    vec3 xTex = texture(tex, xUV).rgb;
    vec3 yTex = texture(tex, yUV).rgb;
    vec3 zTex = texture(tex, zUV).rgb;

    return xTex * blend.x + yTex * blend.y + zTex * blend.z;
}

// ---------------- TERRAIN BLENDING ----------------
vec3 computeTerrainColor(vec3 normal, vec3 worldPos, vec2 uv) {
    float height = worldPos.y;
    float slope = 1.0 - normal.y; 

    // Calculate weights
    float rock_w = smoothstep(u_rock_slope_start, u_rock_slope_end, slope);
    float snow_w = smoothstep(u_snow_height - 5.0, u_snow_height + 5.0, height);
    
    // Sample terrain layers
    vec3 grass = texture(tex_grass, uv).rgb * u_grass_adj; 
    vec3 rock  = triplanar(tex_rock, worldPos, normal, 0.2);
    vec3 snow  = texture(tex_snow, uv).rgb;

    // Linear blending
    vec3 terrain = mix(grass, rock, rock_w);
    terrain = mix(terrain, snow, snow_w);

    return terrain;
}

// ---------------- MAIN ----------------
void main() {
    // 1. NORMAL CALCULATION
    vec3 normal_map = texture(texture2, v_uv).rgb;
    vec3 N;
    if(length(normal_map) < 0.1) {
        N = normalize(v_normal);
    } else {
        normal_map = normal_map * 2.0 - 1.0;
        normal_map.y = -normal_map.y;
        N = normalize(TBN * normal_map);
    }

    vec3 V = normalize(cam_pos - v_world_pos);

    // 2. ALBEDO (TOGGLE LOGIC)
    vec3 albedo;
    if (u_is_terrain) {
        // Use the complex terrain blending
        vec3 terrainColor = computeTerrainColor(N, v_world_pos, v_uv);
        albedo = pow(terrainColor, vec3(2.2)); // Linearize
    } else {
        if (u_use_color) {
            // Use solid color
            albedo = pow(u_color, vec3(2.2)); // Linearize
        } else {
            // Use standard object texture
            vec3 texColor = texture(texture0, v_uv).rgb;
            albedo = pow(texColor, vec3(2.2)); // Linearize
        }
    }

    // 3. MATERIAL PROPERTIES (MRA)
    vec4 mra = texture(texture1, v_uv);
    float metallic = mra.r;
    float roughness = remapRoughness(mra.g);
    float ao = mra.b;

    // 4. LIGHTING PREP
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 Lo = vec3(0.0);

    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(v_world_pos, 1.0);
    float shadowAmount = ShadowCalculation(fragPosLightSpace, N, normalize(-lights[0].direction));

    // 5. LIGHT LOOP
    for(int i = 0; i < light_count; i++) {
        vec3 L;
        float attenuation = 1.0;

        if(lights[i].type == LIGHT_DIRECTIONAL) {
            L = normalize(-lights[i].direction);
        } else {
            vec3 toLight = lights[i].position - v_world_pos;
            float dist = length(toLight);
            L = normalize(toLight);
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        }

        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0001);

        vec3 radiance = lights[i].color * lights[i].intensity * attenuation;

        float D = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, L), 0.0), F0);

        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);

        vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

        float shadowFactor = (i == 0) ? (1.0 - shadowAmount) : 1.0;

        Lo += shadowFactor * (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // 6. FINAL ASSEMBLY
    vec3 ambient = vec3(0.15) * albedo * ao;
    vec3 finalResult = ambient + Lo;

    // HDR Tone Mapping
    finalResult = ACESToneMapping(finalResult);

    // Gamma Correction
    finalResult = pow(finalResult, vec3(1.0 / 2.2)); 

    f_color = vec4(finalResult, 1.0);
}