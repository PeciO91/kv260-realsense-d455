#include "yolov26_realsense_core.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <vart/runner_ext.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

namespace yolov26 {

bool is_native_color_format(rs2_format format)
{
    return format == RS2_FORMAT_YUYV
        || format == RS2_FORMAT_UYVY;
}

void print_supported_profiles()
{
    rs2::context context;
    const rs2::device_list devices = context.query_devices();

    if (devices.size() == 0) {
        throw std::runtime_error("No RealSense device found");
    }

    for (const rs2::device& device : devices) {
        std::cout
            << "Device: "
            << device.get_info(RS2_CAMERA_INFO_NAME)
            << std::endl;

        for (const rs2::sensor& sensor : device.query_sensors()) {
            for (const rs2::stream_profile& profile
                 : sensor.get_stream_profiles()) {
                const rs2_stream stream = profile.stream_type();

                if (stream != RS2_STREAM_COLOR
                    && stream != RS2_STREAM_DEPTH) {
                    continue;
                }

                if ((stream == RS2_STREAM_COLOR
                     && !is_native_color_format(profile.format()))
                    || (stream == RS2_STREAM_DEPTH
                        && profile.format() != RS2_FORMAT_Z16)) {
                    continue;
                }

                const auto video =
                    profile.as<rs2::video_stream_profile>();

                if (!video) {
                    continue;
                }

                std::cout
                    << (stream == RS2_STREAM_COLOR
                        ? "Color "
                        : "Depth ")
                    << video.width()
                    << "x"
                    << video.height()
                    << " @ "
                    << profile.fps()
                    << " FPS "
                    << rs2_format_to_string(profile.format())
                    << std::endl;
            }
        }
    }
}

bool is_lower_resolution_profile(
    const CameraStreamConfiguration& candidate,
    const CameraStreamConfiguration& current)
{
    const int64_t candidate_pixels =
        static_cast<int64_t>(candidate.width) * candidate.height;
    const int64_t current_pixels =
        static_cast<int64_t>(current.width) * current.height;

    if (candidate_pixels != current_pixels) {
        return candidate_pixels < current_pixels;
    }

    if (candidate.width != current.width) {
        return candidate.width < current.width;
    }

    return candidate.height < current.height;
}

CameraConfiguration select_camera_configuration(
    const InferenceOptions& options)
{
    rs2::context context;
    const rs2::device_list devices = context.query_devices();

    if (devices.size() == 0) {
        throw std::runtime_error("No RealSense device found");
    }

    const rs2::device device = devices[0];
    CameraConfiguration selected;
    bool color_found = false;

    for (const rs2::sensor& sensor : device.query_sensors()) {
        for (const rs2::stream_profile& profile
             : sensor.get_stream_profiles()) {
            if (profile.stream_type() != RS2_STREAM_COLOR
                || !is_native_color_format(profile.format())) {
                continue;
            }

            const auto video =
                profile.as<rs2::video_stream_profile>();

            if (!video) {
                continue;
            }

            const CameraStreamConfiguration candidate = {
                video.width(),
                video.height(),
                profile.fps(),
                profile.format()
            };

            if (candidate.width != DEFAULT_COLOR_WIDTH
                || candidate.height != DEFAULT_COLOR_HEIGHT
                || candidate.fps != DEFAULT_COLOR_FPS) {
                continue;
            }

            selected.color = candidate;
            color_found = true;
        }
    }

    if (!color_found) {
        throw std::runtime_error(
            "Requested native RealSense color profile is unavailable"
        );
    }

    bool depth_found = false;

    for (const rs2::sensor& sensor : device.query_sensors()) {
        for (const rs2::stream_profile& profile
             : sensor.get_stream_profiles()) {
            if (profile.stream_type() != RS2_STREAM_DEPTH
                || profile.format() != RS2_FORMAT_Z16
                || profile.fps() != selected.color.fps) {
                continue;
            }

            const auto video =
                profile.as<rs2::video_stream_profile>();

            if (!video) {
                continue;
            }

            if (options.depth_width != 0
                && (video.width() != options.depth_width
                    || video.height() != options.depth_height)) {
                continue;
            }

            const CameraStreamConfiguration candidate = {
                video.width(),
                video.height(),
                profile.fps(),
                profile.format()
            };

            if (!depth_found
                || (options.depth_width == 0
                    && is_lower_resolution_profile(
                        candidate,
                        selected.depth
                    ))) {
                selected.depth = candidate;
                depth_found = true;
            }
        }
    }

    if (!depth_found) {
        throw std::runtime_error(
            "No Z16 depth profile matches the selected color FPS"
        );
    }

    return selected;
}

void print_camera_configuration(
    const CameraConfiguration& configuration)
{
    std::cout
        << "Selected D455 configuration:"
        << std::endl
        << "RGB:   "
        << configuration.color.width
        << "x"
        << configuration.color.height
        << " @ "
        << configuration.color.fps
        << " FPS (native "
        << rs2_format_to_string(configuration.color.native_format)
        << ", OpenCV BGR8)"
        << std::endl
        << "Depth: "
        << configuration.depth.width
        << "x"
        << configuration.depth.height
        << " @ "
        << configuration.depth.fps
        << " FPS "
        << rs2_format_to_string(configuration.depth.native_format)
        << std::endl;
}

DepthMappingCalibration create_depth_mapping_calibration(
    const rs2::pipeline_profile& profile,
    float depth_scale)
{
    const auto color_profile = profile
        .get_stream(RS2_STREAM_COLOR)
        .as<rs2::video_stream_profile>();
    const auto depth_profile = profile
        .get_stream(RS2_STREAM_DEPTH)
        .as<rs2::video_stream_profile>();

    if (!color_profile || !depth_profile || depth_scale <= 0.0f) {
        throw std::runtime_error("Invalid RealSense depth calibration");
    }

    DepthMappingCalibration calibration;
    calibration.color_intrinsics = color_profile.get_intrinsics();
    calibration.depth_intrinsics = depth_profile.get_intrinsics();
    calibration.color_to_depth = color_profile.get_extrinsics_to(depth_profile);
    calibration.depth_to_color = depth_profile.get_extrinsics_to(color_profile);
    calibration.depth_scale = depth_scale;
    calibration.depth_min_m = std::max(
        DEPTH_SEARCH_MIN_METERS,
        depth_scale
    );
    calibration.depth_max_m = std::min(
        DEPTH_SEARCH_MAX_METERS,
        static_cast<float>(std::numeric_limits<uint16_t>::max())
            * depth_scale
    );
    return calibration;
}

class YoloRunner::Impl {
public:
    explicit Impl(const std::string& model_path)
    {
        graph_ = xir::Graph::deserialize(model_path);

        if (!graph_) {
            throw std::runtime_error(
                "Failed to deserialize model: " + model_path
            );
        }

        const xir::Subgraph* root = graph_->get_root_subgraph();

        if (!root) {
            throw std::runtime_error("Model has no root subgraph");
        }

        std::vector<const xir::Subgraph*> dpu_subgraphs;

        for (const xir::Subgraph* child : root->get_children()) {
            if (!child->has_attr("device")) {
                continue;
            }

            std::string device =
                child->get_attr<std::string>("device");
            std::transform(
                device.begin(),
                device.end(),
                device.begin(),
                [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                }
            );

            if (device == "dpu") {
                dpu_subgraphs.push_back(child);
            }
        }

        if (dpu_subgraphs.size() != 1) {
            throw std::runtime_error(
                "Expected exactly one DPU subgraph, found "
                + std::to_string(dpu_subgraphs.size())
            );
        }

        attrs_ = xir::Attrs::create();
        runner_ = vart::RunnerExt::create_runner(
            dpu_subgraphs[0],
            attrs_.get()
        );

        if (!runner_) {
            throw std::runtime_error(
                "Failed to create DPU runner"
            );
        }

        input_buffers_ = runner_->get_inputs();
        output_buffers_ = runner_->get_outputs();

        validate_input();
        validate_outputs();
        build_quantization_lut();
        report_tensor_metadata();
    }

    int input_width() const
    {
        return input_width_;
    }

    int input_height() const
    {
        return input_height_;
    }

    void preprocess(
        const cv::Mat& image,
        LetterboxTransform& transform)
    {
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

        const float gain = std::min(
            input_height_ / static_cast<float>(rgb.rows),
            input_width_ / static_cast<float>(rgb.cols)
        );
        const int resized_w =
            static_cast<int>(std::round(rgb.cols * gain));
        const int resized_h =
            static_cast<int>(std::round(rgb.rows * gain));
        const int pad_w = input_width_ - resized_w;
        const int pad_h = input_height_ - resized_h;
        const int left =
            static_cast<int>(std::round(pad_w / 2.0f - 0.1f));
        const int right =
            static_cast<int>(std::round(pad_w / 2.0f + 0.1f));
        const int top =
            static_cast<int>(std::round(pad_h / 2.0f - 0.1f));
        const int bottom =
            static_cast<int>(std::round(pad_h / 2.0f + 0.1f));

        cv::Mat resized;

        if (rgb.cols != resized_w || rgb.rows != resized_h) {
            cv::resize(
                rgb,
                resized,
                cv::Size(resized_w, resized_h),
                0.0,
                0.0,
                cv::INTER_LINEAR
            );
        } else {
            resized = rgb;
        }

        cv::Mat letterboxed;
        cv::copyMakeBorder(
            resized,
            letterboxed,
            top,
            bottom,
            left,
            right,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114)
        );

        if (letterboxed.cols != input_width_
            || letterboxed.rows != input_height_) {
            throw std::runtime_error(
                "Letterbox output does not match model input"
            );
        }

        const std::size_t pixels =
            static_cast<std::size_t>(input_width_)
            * input_height_
            * 3;

        if (!letterboxed.isContinuous()) {
            letterboxed = letterboxed.clone();
        }

        const auto input_data =
            input_buffers_[0]->data({0, 0, 0, 0});

        if (input_data.first == 0 || input_data.second < pixels) {
            throw std::runtime_error(
                "Model input buffer is too small"
            );
        }

        auto* destination =
            reinterpret_cast<int8_t*>(input_data.first);

        for (std::size_t i = 0; i < pixels; ++i) {
            destination[i] = quantization_lut_[letterboxed.data[i]];
        }

        input_buffers_[0]->sync_for_write(
            0,
            input_buffers_[0]->get_tensor()->get_data_size()
        );

        transform.gain = gain;
        transform.pad_x = left;
        transform.pad_y = top;
    }

    void execute()
    {
        const std::pair<uint32_t, int> job =
            runner_->execute_async(input_buffers_, output_buffers_);
        const int wait_status =
            runner_->wait(static_cast<int>(job.first), -1);

        if (job.second != 0 || wait_status != 0) {
            throw std::runtime_error(
                "DPU execution failed (launch="
                + std::to_string(job.second)
                + ", wait="
                + std::to_string(wait_status)
                + ")"
            );
        }
    }

    std::vector<Detection> decode(
        float confidence_threshold,
        int max_detections) const
    {
        for (vart::TensorBuffer* buffer : output_buffers_) {
            buffer->sync_for_read(
                0,
                buffer->get_tensor()->get_data_size()
            );
        }

        std::vector<Detection> detections;
        const bool keep_all = confidence_threshold <= 0.0f;
        const bool keep_none = confidence_threshold >= 1.0f;
        const double logit_threshold =
            (!keep_all && !keep_none)
                ? std::log(
                    static_cast<double>(confidence_threshold)
                    / (1.0 - confidence_threshold)
                )
                : 0.0;

        for (const OutputLevel& level : levels_) {
            if (keep_none) {
                break;
            }

            const double quantized_threshold =
                logit_threshold / level.class_scale;

            for (int y = 0; y < level.height; ++y) {
                for (int x = 0; x < level.width; ++x) {
                    const std::size_t cell =
                        static_cast<std::size_t>(y) * level.width + x;
                    const int8_t* class_row =
                        level.class_data + cell * NUM_CLASSES;

                    int best_id = 0;
                    int best_q = class_row[0];

                    for (int c = 1; c < NUM_CLASSES; ++c) {
                        if (class_row[c] > best_q) {
                            best_q = class_row[c];
                            best_id = c;
                        }
                    }

                    if (!keep_all
                        && best_q <= quantized_threshold) {
                        continue;
                    }

                    const float logit =
                        static_cast<float>(best_q)
                        * level.class_scale;
                    const float score = logit >= 0.0f
                        ? 1.0f / (1.0f + std::exp(-logit))
                        : std::exp(logit)
                            / (1.0f + std::exp(logit));

                    const int8_t* box_row =
                        level.box_data + cell * 4 * REG_MAX;
                    const float anchor_x = x + 0.5f;
                    const float anchor_y = y + 0.5f;
                    const float stride =
                        static_cast<float>(level.stride);

                    Detection detection;
                    detection.class_id = best_id;
                    detection.score = score;
                    detection.x1 =
                        (anchor_x - box_row[0] * level.box_scale)
                        * stride;
                    detection.y1 =
                        (anchor_y - box_row[1] * level.box_scale)
                        * stride;
                    detection.x2 =
                        (anchor_x + box_row[2] * level.box_scale)
                        * stride;
                    detection.y2 =
                        (anchor_y + box_row[3] * level.box_scale)
                        * stride;
                    detections.push_back(detection);
                }
            }
        }

        std::stable_sort(
            detections.begin(),
            detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.score > b.score;
            }
        );

        if (detections.size()
            > static_cast<std::size_t>(max_detections)) {
            detections.resize(
                static_cast<std::size_t>(max_detections)
            );
        }

        return detections;
    }

private:
    struct OutputLevel {
        int stride = 0;
        int height = 0;
        int width = 0;
        float box_scale = 0.0f;
        float class_scale = 0.0f;
        const int8_t* box_data = nullptr;
        const int8_t* class_data = nullptr;
    };

    void validate_input()
    {
        const std::vector<const xir::Tensor*> inputs =
            runner_->get_input_tensors();

        if (inputs.size() != 1) {
            throw std::runtime_error(
                "Expected exactly one model input, got "
                + std::to_string(inputs.size())
            );
        }

        const xir::Tensor* tensor = inputs[0];
        const std::vector<int32_t> shape = tensor->get_shape();

        if (shape.size() != 4
            || shape[0] != 1
            || shape[3] != 3) {
            throw std::runtime_error(
                "Model input must be rank-4 batch-one NHWC "
                "with 3 channels"
            );
        }

        const xir::DataType data_type = tensor->get_data_type();

        if ((data_type.type != xir::DataType::INT
             && data_type.type != xir::DataType::XINT)
            || data_type.bit_width != 8) {
            throw std::runtime_error(
                "Model input must be INT8"
            );
        }

        if (!tensor->has_attr("fix_point")) {
            throw std::runtime_error(
                "Model input is missing fix_point metadata"
            );
        }

        input_fix_point_ = tensor->get_attr<int>("fix_point");
        input_height_ = shape[1];
        input_width_ = shape[2];
    }

    void validate_outputs()
    {
        const std::vector<const xir::Tensor*> outputs =
            runner_->get_output_tensors();

        if (outputs.size() != 2 * STRIDES.size()) {
            throw std::runtime_error(
                "Expected "
                + std::to_string(2 * STRIDES.size())
                + " model outputs, got "
                + std::to_string(outputs.size())
            );
        }

        struct Group {
            int height = 0;
            int width = 0;
            int box_index = -1;
            int class_index = -1;
        };

        std::map<std::pair<int, int>, Group> groups;

        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const xir::Tensor* tensor = outputs[i];
            const std::vector<int32_t> shape =
                tensor->get_shape();

            if (shape.size() != 4 || shape[0] != 1) {
                throw std::runtime_error(
                    "Model output "
                    + tensor->get_name()
                    + " must be rank-4 batch-one NHWC"
                );
            }

            const xir::DataType data_type =
                tensor->get_data_type();

            if ((data_type.type != xir::DataType::INT
                 && data_type.type != xir::DataType::XINT)
                || data_type.bit_width != 8) {
                throw std::runtime_error(
                    "Model output "
                    + tensor->get_name()
                    + " must be INT8"
                );
            }

            if (!tensor->has_attr("fix_point")) {
                throw std::runtime_error(
                    "Model output "
                    + tensor->get_name()
                    + " is missing fix_point metadata"
                );
            }

            const int channels = shape[3];
            Group& group = groups[{shape[1], shape[2]}];
            group.height = shape[1];
            group.width = shape[2];

            if (channels == 4 * REG_MAX) {
                if (group.box_index >= 0) {
                    throw std::runtime_error(
                        "Duplicate box output at spatial shape "
                        + std::to_string(shape[1])
                        + "x"
                        + std::to_string(shape[2])
                    );
                }
                group.box_index = static_cast<int>(i);
            } else if (channels == NUM_CLASSES) {
                if (group.class_index >= 0) {
                    throw std::runtime_error(
                        "Duplicate class output at spatial shape "
                        + std::to_string(shape[1])
                        + "x"
                        + std::to_string(shape[2])
                    );
                }
                group.class_index = static_cast<int>(i);
            } else {
                throw std::runtime_error(
                    "Model output "
                    + tensor->get_name()
                    + " has "
                    + std::to_string(channels)
                    + " channels; expected "
                    + std::to_string(4 * REG_MAX)
                    + " (box) or "
                    + std::to_string(NUM_CLASSES)
                    + " (class)"
                );
            }
        }

        if (groups.size() != STRIDES.size()) {
            throw std::runtime_error(
                "Expected "
                + std::to_string(STRIDES.size())
                + " distinct output spatial shapes, got "
                + std::to_string(groups.size())
            );
        }

        std::vector<Group> ordered;

        for (const auto& entry : groups) {
            const Group& group = entry.second;

            if (group.box_index < 0 || group.class_index < 0) {
                throw std::runtime_error(
                    "Malformed detection level at spatial shape "
                    + std::to_string(group.height)
                    + "x"
                    + std::to_string(group.width)
                );
            }

            ordered.push_back(group);
        }

        std::stable_sort(
            ordered.begin(),
            ordered.end(),
            [](const Group& a, const Group& b) {
                return a.height * a.width > b.height * b.width;
            }
        );

        for (std::size_t level = 0; level < ordered.size(); ++level) {
            const Group& group = ordered[level];
            const int stride = STRIDES[level];

            if (group.height != input_height_ / stride
                || group.width != input_width_ / stride) {
                throw std::runtime_error(
                    "Detection level for stride "
                    + std::to_string(stride)
                    + " has spatial shape "
                    + std::to_string(group.height)
                    + "x"
                    + std::to_string(group.width)
                    + "; expected "
                    + std::to_string(input_height_ / stride)
                    + "x"
                    + std::to_string(input_width_ / stride)
                );
            }

            const xir::Tensor* box_tensor =
                outputs[group.box_index];
            const xir::Tensor* class_tensor =
                outputs[group.class_index];

            OutputLevel output_level;
            output_level.stride = stride;
            output_level.height = group.height;
            output_level.width = group.width;
            output_level.box_scale = static_cast<float>(
                std::exp2(
                    -box_tensor->get_attr<int>("fix_point")
                )
            );
            output_level.class_scale = static_cast<float>(
                std::exp2(
                    -class_tensor->get_attr<int>("fix_point")
                )
            );

            const auto box_data =
                output_buffers_[group.box_index]
                    ->data({0, 0, 0, 0});
            const std::size_t box_bytes =
                static_cast<std::size_t>(group.height)
                * group.width
                * 4
                * REG_MAX;

            if (box_data.first == 0 || box_data.second < box_bytes) {
                throw std::runtime_error(
                    "Model output buffer for "
                    + box_tensor->get_name()
                    + " is too small"
                );
            }

            const auto class_data =
                output_buffers_[group.class_index]
                    ->data({0, 0, 0, 0});
            const std::size_t class_bytes =
                static_cast<std::size_t>(group.height)
                * group.width
                * NUM_CLASSES;

            if (class_data.first == 0
                || class_data.second < class_bytes) {
                throw std::runtime_error(
                    "Model output buffer for "
                    + class_tensor->get_name()
                    + " is too small"
                );
            }

            output_level.box_data =
                reinterpret_cast<const int8_t*>(box_data.first);
            output_level.class_data =
                reinterpret_cast<const int8_t*>(class_data.first);
            levels_.push_back(output_level);
        }
    }

    void build_quantization_lut()
    {
        const double scale = std::exp2(
            static_cast<double>(input_fix_point_)
        );

        for (int i = 0; i < 256; ++i) {
            const long quantized = std::lround(
                i * scale / 255.0
            );
            quantization_lut_[i] = static_cast<int8_t>(
                std::clamp(quantized, -128L, 127L)
            );
        }
    }

    void report_tensor_metadata() const
    {
        const auto describe = [](const xir::Tensor* tensor) {
            std::ostringstream out;
            out << "  " << tensor->get_name() << " shape=[";

            const std::vector<int32_t> shape = tensor->get_shape();

            for (std::size_t i = 0; i < shape.size(); ++i) {
                out << (i == 0 ? "" : ",") << shape[i];
            }

            out << "] "
                << tensor->get_data_type().to_string();

            if (tensor->has_attr("fix_point")) {
                out << " fix_point="
                    << tensor->get_attr<int>("fix_point");
            }

            return out.str();
        };

        std::cout << "Model input tensors:" << std::endl;

        for (const xir::Tensor* tensor
             : runner_->get_input_tensors()) {
            std::cout << describe(tensor) << std::endl;
        }

        std::cout << "Model output tensors:" << std::endl;

        for (const xir::Tensor* tensor
             : runner_->get_output_tensors()) {
            std::cout << describe(tensor) << std::endl;
        }
    }

    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<xir::Attrs> attrs_;
    std::unique_ptr<vart::RunnerExt> runner_;
    std::vector<vart::TensorBuffer*> input_buffers_;
    std::vector<vart::TensorBuffer*> output_buffers_;
    std::vector<OutputLevel> levels_;
    std::array<int8_t, 256> quantization_lut_{};
    int input_width_ = 0;
    int input_height_ = 0;
    int input_fix_point_ = 0;
};

YoloRunner::YoloRunner(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path))
{
}

YoloRunner::~YoloRunner() = default;

int YoloRunner::input_width() const
{
    return impl_->input_width();
}

int YoloRunner::input_height() const
{
    return impl_->input_height();
}

void YoloRunner::preprocess(
    const cv::Mat& image,
    LetterboxTransform& transform)
{
    impl_->preprocess(image, transform);
}

void YoloRunner::execute()
{
    impl_->execute();
}

std::vector<Detection> YoloRunner::decode(
    float confidence_threshold,
    int max_detections) const
{
    return impl_->decode(confidence_threshold, max_detections);
}

float median_depth_in_color_roi(
    const rs2::depth_frame& depth,
    const DepthMappingCalibration& calibration,
    int x1,
    int y1,
    int x2,
    int y2)
{
    x1 = std::clamp(x1, 0, calibration.color_intrinsics.width - 1);
    y1 = std::clamp(y1, 0, calibration.color_intrinsics.height - 1);
    x2 = std::clamp(x2, 0, calibration.color_intrinsics.width - 1);
    y2 = std::clamp(y2, 0, calibration.color_intrinsics.height - 1);

    if (x2 <= x1 || y2 <= y1
        || depth.get_width() != calibration.depth_intrinsics.width
        || depth.get_height() != calibration.depth_intrinsics.height) {
        return 0.0f;
    }

    const auto* data = reinterpret_cast<const uint16_t*>(depth.get_data());
    std::array<
        uint16_t,
        DEPTH_SAMPLE_GRID_SIZE * DEPTH_SAMPLE_GRID_SIZE
    > values{};
    std::size_t value_count = 0;
    const float roi_width = static_cast<float>(x2 - x1);
    const float roi_height = static_cast<float>(y2 - y1);

    for (int row = 0; row < DEPTH_SAMPLE_GRID_SIZE; ++row) {
        for (int column = 0; column < DEPTH_SAMPLE_GRID_SIZE; ++column) {
            const float color_pixel[2] = {
                x1 + (column + 0.5f) * roi_width
                    / DEPTH_SAMPLE_GRID_SIZE,
                y1 + (row + 0.5f) * roi_height
                    / DEPTH_SAMPLE_GRID_SIZE
            };
            float depth_pixel[2] = {-1.0f, -1.0f};

            rs2_project_color_pixel_to_depth_pixel(
                depth_pixel,
                data,
                calibration.depth_scale,
                calibration.depth_min_m,
                calibration.depth_max_m,
                &calibration.depth_intrinsics,
                &calibration.color_intrinsics,
                &calibration.color_to_depth,
                &calibration.depth_to_color,
                color_pixel
            );

            if (!std::isfinite(depth_pixel[0])
                || !std::isfinite(depth_pixel[1])) {
                continue;
            }

            const int depth_x = static_cast<int>(depth_pixel[0]);
            const int depth_y = static_cast<int>(depth_pixel[1]);

            if (depth_x < 0 || depth_x >= depth.get_width()
                || depth_y < 0 || depth_y >= depth.get_height()) {
                continue;
            }

            const uint16_t raw = data[
                depth_y * calibration.depth_intrinsics.width + depth_x
            ];

            if (raw != 0) {
                values[value_count++] = raw;
            }
        }
    }

    if (value_count == 0) {
        return 0.0f;
    }

    auto middle = values.begin() + value_count / 2;
    std::nth_element(values.begin(), middle, values.begin() + value_count);
    return static_cast<float>(*middle) * calibration.depth_scale;
}

void draw_detection_annotations(
    cv::Mat& image,
    const std::vector<DetectionAnnotation>& annotations,
    const InferenceOptions& options)
{
    for (const DetectionAnnotation& detection : annotations) {
        const cv::Scalar box_color =
            detection.class_id == 0
                ? cv::Scalar(0, 255, 0)
                : cv::Scalar(0, 165, 255);

        cv::rectangle(
            image,
            cv::Rect(
                detection.x,
                detection.y,
                detection.width,
                detection.height
            ),
            box_color,
            2
        );

        if (options.show_depth_roi) {
            cv::rectangle(
                image,
                cv::Point(
                    std::max(0, detection.depth_x1),
                    std::max(0, detection.depth_y1)
                ),
                cv::Point(
                    std::min(
                        image.cols - 1,
                        detection.depth_x2
                    ),
                    std::min(
                        image.rows - 1,
                        detection.depth_y2
                    )
                ),
                cv::Scalar(255, 0, 0),
                1
            );
        }

        char label[128];
        const char* class_name =
            detection.class_id >= 0
                && detection.class_id < NUM_CLASSES
                ? CLASS_NAMES[detection.class_id]
                : "Unknown";

        if (detection.distance > 0.0f) {
            if (options.show_confidence) {
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s %.2f | %.2f m",
                    class_name,
                    detection.score,
                    detection.distance
                );
            } else {
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s | %.2f m",
                    class_name,
                    detection.distance
                );
            }
        } else {
            if (options.show_confidence) {
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s %.2f | N/A",
                    class_name,
                    detection.score
                );
            } else {
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s | N/A",
                    class_name
                );
            }
        }

        cv::putText(
            image,
            label,
            cv::Point(
                detection.x,
                std::max(20, detection.y - 8)
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv::LINE_AA
        );
    }
}

bool read_text_file(
    const std::string& path,
    std::string& value)
{
    std::ifstream input(path);

    if (!input) {
        return false;
    }

    return static_cast<bool>(std::getline(input, value));
}

bool read_number_file(
    const std::string& path,
    double& value)
{
    std::ifstream input(path);
    return static_cast<bool>(input >> value);
}

void SystemMonitor::discover_hwmon()
{
    DIR* directory = opendir("/sys/class/hwmon");

    if (!directory) {
        return;
    }

    while (dirent* entry = readdir(directory)) {
        const std::string entry_name(entry->d_name);

        if (entry_name.compare(0, 5, "hwmon") != 0) {
            continue;
        }

        const std::string path =
            "/sys/class/hwmon/" + entry_name;
        std::string name;

        if (!read_text_file(path + "/name", name)) {
            continue;
        }

        if (name == "ams") {
            ams_path_ = path;
        } else if (name.compare(0, 6, "ina260") == 0) {
            ina260_path_ = path;
        }
    }

    closedir(directory);
}

void SystemMonitor::update_cpu()
{
    std::ifstream input("/proc/stat");
    std::string label;
    CpuTimes current;

    if (!(input
          >> label
          >> current.user
          >> current.nice
          >> current.system
          >> current.idle
          >> current.iowait
          >> current.irq
          >> current.softirq
          >> current.steal)
        || label != "cpu") {
        stats_.cpu_available = false;
        return;
    }

    stats_.cpu_available = false;

    if (has_cpu_sample_) {
        const uint64_t current_total = current.total();
        const uint64_t previous_total = previous_cpu_.total();
        const uint64_t current_idle = current.idle_total();
        const uint64_t previous_idle = previous_cpu_.idle_total();

        if (current_total > previous_total
            && current_idle >= previous_idle) {
            const uint64_t total_delta =
                current_total - previous_total;
            const uint64_t idle_delta =
                current_idle - previous_idle;

            if (idle_delta <= total_delta) {
                stats_.cpu_percent =
                    100.0
                    * static_cast<double>(
                        total_delta - idle_delta
                    )
                    / static_cast<double>(total_delta);
                stats_.cpu_available = true;
            }
        }
    }

    previous_cpu_ = current;
    has_cpu_sample_ = true;
}

void SystemMonitor::update_memory()
{
    std::ifstream input("/proc/meminfo");
    std::string line;
    uint64_t total_kb = 0;
    uint64_t available_kb = 0;

    while (std::getline(input, line)) {
        std::istringstream values(line);
        std::string key;
        uint64_t value = 0;

        if (!(values >> key >> value)) {
            continue;
        }

        if (key == "MemTotal:") {
            total_kb = value;
        } else if (key == "MemAvailable:") {
            available_kb = value;
        }
    }

    stats_.ram_available =
        total_kb > 0 && available_kb <= total_kb;

    if (!stats_.ram_available) {
        return;
    }

    const uint64_t used_kb = total_kb - available_kb;
    stats_.ram_used_mb = used_kb / 1024;
    stats_.ram_total_mb = total_kb / 1024;
    stats_.ram_percent =
        100.0
        * static_cast<double>(used_kb)
        / static_cast<double>(total_kb);
}

void SystemMonitor::update_sensors()
{
    stats_.temperature_available = false;
    stats_.pl_temperature_available = false;
    stats_.power_available = false;
    stats_.voltage_available = false;
    stats_.current_available = false;

    if (!ams_path_.empty()) {
        double ps_lpd = 0.0;
        double ps_fpd = 0.0;
        double pl = 0.0;
        const bool has_ps_lpd = read_number_file(
            ams_path_ + "/temp1_input",
            ps_lpd
        );
        const bool has_ps_fpd = read_number_file(
            ams_path_ + "/temp2_input",
            ps_fpd
        );

        if (has_ps_lpd || has_ps_fpd) {
            const double ps_temperature =
                has_ps_lpd && has_ps_fpd
                ? std::max(ps_lpd, ps_fpd)
                : (has_ps_lpd ? ps_lpd : ps_fpd);
            stats_.temperature_c = ps_temperature / 1000.0;
            stats_.temperature_available = true;
        }

        if (read_number_file(
                ams_path_ + "/temp3_input",
                pl)) {
            stats_.pl_temperature_c = pl / 1000.0;
            stats_.pl_temperature_available = true;
        }
    }

    if (!ina260_path_.empty()) {
        double power_uw = 0.0;
        double voltage_mv = 0.0;
        double current_ma = 0.0;

        if (read_number_file(
                ina260_path_ + "/power1_input",
                power_uw)) {
            stats_.power_w = power_uw / 1000000.0;
            stats_.power_available = true;
        }

        if (read_number_file(
                ina260_path_ + "/in1_input",
                voltage_mv)) {
            stats_.voltage_v = voltage_mv / 1000.0;
            stats_.voltage_available = true;
        }

        if (read_number_file(
                ina260_path_ + "/curr1_input",
                current_ma)) {
            stats_.current_a = current_ma / 1000.0;
            stats_.current_available = true;
        }
    }
}

std::optional<FramePacket> infer_frame(
    const rs2::frameset& frames,
    const CameraConfiguration& camera_configuration,
    YoloRunner& detector,
    const DepthMappingCalibration& depth_mapping,
    const InferenceOptions& options,
    uint64_t sequence,
    Clock::time_point frame_start,
    Clock::time_point acquire_end)
{
    auto color = frames.get_color_frame();
    auto depth = frames.get_depth_frame();

    if (!color || !depth) {
        return std::nullopt;
    }

    cv::Mat image(
        cv::Size(
            camera_configuration.color.width,
            camera_configuration.color.height
        ),
        CV_8UC3,
        const_cast<void*>(
            color.get_data()
        ),
        cv::Mat::AUTO_STEP
    );

    image = image.clone();

    LetterboxTransform letterbox;
    detector.preprocess(image, letterbox);

    const auto preprocess_end = Clock::now();

    detector.execute();

    const auto dpu_end = Clock::now();

    const std::vector<Detection> detections =
        detector.decode(
            options.confidence_threshold,
            options.max_detections
        );

    std::vector<DetectionAnnotation> annotations;
    annotations.reserve(16);

    if (annotations.capacity() < detections.size()) {
        annotations.reserve(detections.size());
    }

    for (const Detection& detection : detections) {
        if (!std::isfinite(detection.score)
            || !std::isfinite(detection.x1)
            || !std::isfinite(detection.y1)
            || !std::isfinite(detection.x2)
            || !std::isfinite(detection.y2)
            || detection.class_id < 0
            || detection.class_id >= NUM_CLASSES
            || !std::isfinite(letterbox.gain)
            || letterbox.gain <= 0.0f) {
            continue;
        }

        const float projected_x1 =
            (detection.x1 - letterbox.pad_x)
            / letterbox.gain;
        const float projected_y1 =
            (detection.y1 - letterbox.pad_y)
            / letterbox.gain;
        const float projected_x2 =
            (detection.x2 - letterbox.pad_x)
            / letterbox.gain;
        const float projected_y2 =
            (detection.y2 - letterbox.pad_y)
            / letterbox.gain;
        const double int_min =
            static_cast<double>(std::numeric_limits<int>::min());
        const double int_max =
            static_cast<double>(std::numeric_limits<int>::max());

        if (!std::isfinite(projected_x1)
            || !std::isfinite(projected_y1)
            || !std::isfinite(projected_x2)
            || !std::isfinite(projected_y2)
            || static_cast<double>(projected_x1) < int_min
            || static_cast<double>(projected_x1) > int_max
            || static_cast<double>(projected_y1) < int_min
            || static_cast<double>(projected_y1) > int_max
            || static_cast<double>(projected_x2) < int_min
            || static_cast<double>(projected_x2) > int_max
            || static_cast<double>(projected_y2) < int_min
            || static_cast<double>(projected_y2) > int_max) {
            continue;
        }

        int x = static_cast<int>(
            (detection.x1 - letterbox.pad_x)
            / letterbox.gain
        );

        int y = static_cast<int>(
            (detection.y1 - letterbox.pad_y)
            / letterbox.gain
        );

        int x2 = static_cast<int>(
            (detection.x2 - letterbox.pad_x)
            / letterbox.gain
        );

        int y2 = static_cast<int>(
            (detection.y2 - letterbox.pad_y)
            / letterbox.gain
        );

        x = std::clamp(x, 0, image.cols - 1);
        y = std::clamp(y, 0, image.rows - 1);
        x2 = std::clamp(x2, 0, image.cols);
        y2 = std::clamp(y2, 0, image.rows);

        int w = x2 - x;

        int h = y2 - y;

        if (w <= 0 || h <= 0) {
            continue;
        }

        int cx = x + w / 2;
        int cy = y + h / 2;

        int roi_w = std::max(
            2,
            static_cast<int>(
                w * DEPTH_ROI_SCALE
            )
        );

        int roi_h = std::max(
            2,
            static_cast<int>(
                h * DEPTH_ROI_SCALE
            )
        );

        int dx1 =
            cx - roi_w / 2;

        int dy1 =
            cy - roi_h / 2;

        int dx2 =
            cx + roi_w / 2;

        int dy2 =
            cy + roi_h / 2;

        const float distance =
            median_depth_in_color_roi(
                depth,
                depth_mapping,
                dx1,
                dy1,
                dx2,
                dy2
            );

        annotations.push_back({
            detection.class_id,
            x,
            y,
            w,
            h,
            dx1,
            dy1,
            dx2,
            dy2,
            detection.score,
            distance
        });
    }

    const auto depth_end = Clock::now();

    FramePacket packet;
    packet.sequence = sequence;
    packet.image = std::move(image);
    packet.annotations = std::move(annotations);
    packet.frame_start = frame_start;
    packet.acquired_at = acquire_end;
    packet.completed_at = depth_end;
    packet.acquire_ms = elapsed_ms(
        frame_start,
        acquire_end
    );
    packet.preprocess_ms = elapsed_ms(
        acquire_end,
        preprocess_end
    );
    packet.dpu_ms = elapsed_ms(
        preprocess_end,
        dpu_end
    );
    packet.depth_map_ms = elapsed_ms(
        dpu_end,
        depth_end
    );

    return packet;
}

}
