//
// Created by gabriel on 11.07.21.
//

#include "reconstruction.h"


bool CheckTriangularValidity(Vertex* vertices, unsigned int one, unsigned int two, unsigned int three, float threshold) {
    // check if all vertices are valid
    if (vertices[one].position.x() == MINF || vertices[two].position.x() == MINF || vertices[three].position.x() == MINF)
    {
        return false;
    }
    // compute length of edges
    float l1, l2, l3;
    l1 = sqrtf(powf(vertices[one].position.x() - vertices[two].position.x(), 2) + \
                  powf(vertices[one].position.y() - vertices[two].position.y(), 2) + \
                  powf(vertices[one].position.z() - vertices[two].position.z(), 2));
    l2 = sqrtf(powf(vertices[one].position.x() - vertices[three].position.x(), 2) + \
                  powf(vertices[one].position.y() - vertices[three].position.y(), 2) + \
                  powf(vertices[one].position.z() - vertices[three].position.z(), 2));
    l3 = sqrtf(powf(vertices[two].position.x() - vertices[three].position.x(), 2) + \
                  powf(vertices[two].position.y() - vertices[three].position.y(), 2) + \
                  powf(vertices[two].position.z() - vertices[three].position.z(), 2));
    // check if length of edges below threshold
    if (l1 > threshold || l2 > threshold || l3 > threshold)
    {
        return false;
    }
    return true;
}


bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename) {
    float edgeThreshold = 0.01f; // 1cm

    // use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
    // - have a look at the "off_sample.off" file to see how to store the vertices and triangles
    // - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
    // - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
    // - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
    // - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
    // - only write triangles with valid vertices and an edge length smaller then edgeThreshold

    // get number of vertices
    unsigned int nVertices = 0;
    nVertices = width*height;

    // determine number of valid faces
    unsigned nFaces = 0;
    // using the same vertex indexing for grid box corners: first dimension height, second dimension width
    // generate valid triangle triplets in a row array counterclockwise (number of grid boxes * 2 triangles * 3 points)
    std::vector<unsigned int> triangles;

    // go through grids
    for (uint h=0; h < height-1; h++)
    {
        for (uint w=0; w < width-1; w++)
        {
            // upper left triangle
            if (CheckTriangularValidity(vertices, h * width + w, (h+1) * width + w, h * width + (w+1), edgeThreshold))
            {
                triangles.emplace_back(h * width + w);
                triangles.emplace_back((h+1) * width + w);
                triangles.emplace_back(h * width + (w+1));
                nFaces++;
            }
            // lower right triangle
            if (CheckTriangularValidity(vertices, (h+1) * width + w, (h+1) * width + (w+1), h * width + (w+1), edgeThreshold))
            {
                triangles.emplace_back((h+1) * width + w);
                triangles.emplace_back((h+1) * width + (w+1));
                triangles.emplace_back(h * width + (w+1));
                nFaces++;
            }
        }
    }

    // write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // save vertices
    for (int n=0; n < nVertices; n++)
    {
        if (vertices[n].position.x() == MINF)
        {
            outFile << "0 0 0 " << (uint) vertices[n].color.x() << " " << (uint) vertices[n].color.y() << " " \
                    << (uint) vertices[n].color.z() << " " << (uint) vertices[n].color.w() << std::endl;
        }
        else
        {
            outFile << vertices[n].position.x() << " " << vertices[n].position.y() << " " << vertices[n].position.z() << " " \
                    << (uint) vertices[n].color.x() << " " << (uint) vertices[n].color.y() << " " \
                    << (uint) vertices[n].color.z() << " " << (uint) vertices[n].color.w() << std::endl;
        }
    }

    // save valid faces
    for (uint n=0; n < nFaces; n++)
    {
        outFile << "3 " << triangles[3 * n] << " " << triangles[3 * n + 1] << " " << triangles[3 * n + 2] << std::endl;
    }

    // close file
    outFile.close();

    return true;
}


void reconstruction(cv::Mat bgrImage, cv::Mat depthValues, Matrix3f intrinsics) {
    // TODO: @Tim: Aktuell werden ungültige Depth Werte als MINF erwartet

    float fX = intrinsics(0, 0);
    float fY = intrinsics(1, 1);
    float cX = intrinsics(0, 2);
    float cY = intrinsics(1, 2);

    // back-projection
    // write result to the vertices array below, keep pixel ordering!
    // if the depth value at idx is invalid (MINF) write the following values to the vertices array
    // vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
    // vertices[idx].color = Vector4uc(0,0,0,0);
    // otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap

    Vertex *vertices = new Vertex[bgrImage.cols * bgrImage.rows];

    float depth;
    unsigned int idx;
    for (uint h=0; h < bgrImage.rows; h++) {
        for (int w = 0; w < bgrImage.cols; w++) {
            // index in vertex array
            idx = h * depthValues.cols + w;

            // check if depth value invalid
            depth = depthValues.at<float>(h, w);
            if (depth == MINF) {
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
            } else {
                // back-project pixels of depth map to camera space
                float x_camera = ((float) w * depth - cX * depth) / fX;
                float y_camera = ((float) h * depth - cY * depth) / fY;

                // transform to world space
                vertices[idx].position = Vector4f(x_camera, y_camera, depth, 1);

                // assign the color information to vertices assuming the same image size
                Vector4uc bgr = bgrImage.at<Vector4uc>(h, w);
                Vector4uc rgb = bgr;
//                rgb(0) = bgr(2);
//                rgb(2) = bgr(0);
                vertices[idx].color = rgb;
            }
        }
    }

    // write mesh file
    std::stringstream ss;
    ss << "../../results/reconstruction_mesh.off";
    if (!WriteMesh(vertices, depthValues.cols, depthValues.rows, ss.str()))
    {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
    }

    // free memory
    delete[] vertices;
}
