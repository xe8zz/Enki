// -- preview --
//
// use enki;
//
// struct Particle {
//     position: [f32; 3],
//     color: [f32; 4],
//     mass: f32,
// }
// #[enki_compute]
// fn update_array(my_particle: Particle, g: f32) {
//     let f = my_particle.mass * g;
//     let a = f / my_particle.mass;
//     let vel += a;
//     my_particle.position += vel;
// }
// #[enki_fragment]
// fn present_array(my_particle: Particle) {
//     return my_particle.color;
// }
// fn main() {
//     const SIZE: usize = 1_000_000;
//     const G: f32 = 9.8;
//
//     let engine = enki::engine::new();
//     let window = enki::window::new(1080, 720, "my engine");
//
//     let particles: [Particle; SIZE];
//     for i in 0..SIZE {
//         let particle = Particle { position: somePositions[i], color: someColors[i], mass: someMasses[i] };
//
//         particles[i] = particle;
//     }
//
//     let array = enki.array<Particle>(particle);
//
//     window.run(|| {
//         update_array(array, G);
//         present_array(array);
//     });
// }
//
